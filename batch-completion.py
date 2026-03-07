#!/usr/bin/env python3
"""Minimal plain-text interface for Anthropic's large language model
"Claude", utilising batch mode for lower inference costs.

The input and output are formatted to match the text in the model's
context window as closely as possible. The conversation history is read
from standard input. The user is expected to maintain it in a text file,
adding their own message at the end, before using this script to request
a response from Claude.

Input must consist of an optional system prompt followed by strictly
alternating "Human" & "Assistant" messages. The last message **must** be
an "Assistant" message (which may be empty) in order for a response to
be requested immediately. If the last message is a "Human" message, the
script will wait for further input to append to the conversation
history. This behaviour is intended to preserve symmetry between the
"Human" and "Assistant" roles: each participant indicates the end of
their message by writing the other participant's role label. Note that
role labels must be preceded by two newline characters and followed by a
single space character. Leading & trailing whitespace will be trimmed
from each message.

Responses are written to standard output. If more than one response is
requested, they will be separated by a "file separator" control
character (U+001C).

Complete messages (`stop_reason = "end_turn"`) will end with the "Human"
turn marker. Incomplete messages will have the `stop_reason` value
appended after an "end of text" character (U+0003). Requests which could
not be completed will be indicated with an "end of text" character
followed by the error message.

Tool use and structured output are not supported by this script.

"""

########################################################################
# IMPORTS

from typing import (
    Final,
    Literal,
    Optional,
)
from collections.abc import (
    Iterable,
    AsyncIterable,
    AsyncIterator,
)
from sys import (
    argv,
    stdin,
    stderr,
)
from argparse import (
    ArgumentParser,
    BooleanOptionalAction,
)
from asyncio import (
    run as asyncio_run,
    sleep as asyncio_sleep,
)
from logging import (
    basicConfig as logging_config,
    DEBUG as LOGGING_LEVEL_DEBUG,
)

from anthropic import (
    AsyncAnthropic,
    APIError as AnthropicAPIError,
)
from anthropic.types.stop_reason import StopReason
from anthropic.types.message import Message
from anthropic.types.message_param import MessageParam
from anthropic.types.text_block import TextBlock
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.thinking_block import ThinkingBlock
from anthropic.types.thinking_block_param import ThinkingBlockParam
from anthropic.types.cache_control_ephemeral_param import CacheControlEphemeralParam
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import (
    MessageBatch,
    MessageBatchResult,
    MessageBatchErroredResult,
    MessageBatchExpiredResult,
    MessageBatchSucceededResult,
    MessageBatchIndividualResponse,
)
from anthropic.types.messages.batch_create_params import (
    Request as MessageBatchCreateRequest,
)
from anthropic.types.shared import ErrorObject as AnthropicErrorObject
from anthropic.types.thinking_config_enabled_param import ThinkingConfigEnabledParam
from anthropic.types.thinking_config_disabled_param import ThinkingConfigDisabledParam

from anthropic_utils import (
    MessageRole,
    MESSAGE_ROLE_USER,
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_LABEL_USER,
    MESSAGE_ROLE_LABEL_ASSISTANT,
    CONTENT_TYPE_TEXT,
    CONTENT_TYPE_THINKING,
    MessageStopReason,
    MESSAGE_STOP_REASON_END_TURN,
    MESSAGE_STOP_REASON_STOP_SEQUENCE,
    MESSAGE_STOP_REASON_REFUSAL,
    MessageBatchResultType,
    MESSAGE_BATCH_RESULT_TYPE_SUCCEEDED,
    MESSAGE_BATCH_RESULT_TYPE_ERRORED,
    message_batch_submit_and_poll_until_completion,
    message_split,
    message_to_str,
    content_to_str,
)

########################################################################
# EXPORTS

__all__ = (
    "batch_completion",
    "main",
)

########################################################################
# CONSTANTS

DELIMITER_COMPLETION: Final[str] = "\x1C"
"""Delimiter character between output completions."""

DELIMITER_STOP_REASON_OR_ERROR: Final[str] = "\x03"
"""Delimiter character before `stop_reason` or an error message."""

DEFAULT_MODEL_ID: Final[str] = "claude-haiku-4-5-20251001"
"""Default model ID."""

DEFAULT_TEMPERATURE: Final[float] = 1.0
"""Default temperature."""

DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST: Final[int] = 32768
"""Default maximum number of output tokens to generate for any single API request."""

DEFAULT_THINKING_BUDGET: Final[int] = 4096
"""Default maximum number of tokens in `thinking` blocks."""

DEFAULT_POLL_INTERVAL: Final[int] = 30
"""Default interval between polling the status of the message batch."""

# Exit statuses
EXIT_SUCCESS: Final[int] = 0
EXIT_ERROR: Final[int] = 1
EXIT_TRUNCATED: Final[int] = 2
EXIT_REFUSAL: Final[int] = 4

########################################################################
# FUNCTIONS

def write_stdout(text: str) -> None:
    """Write `text` to standard output, followed by a newline, and flush the stream."""
    print(text, flush=True)

def write_stderr(text: str) -> None:
    """Write `text` to standard error, followed by a newline, and flush the stream."""
    print(text, file=stderr, flush=True)

# def split_trailing_whitespace(text: str) -> tuple[str, str]:
#     """Strip trailing whitespace from `text` (via `rstrip()`) and return `(rstripped_text, whitespace)`."""
#     rstripped_text = text.rstrip()
#     whitespace = text[len(rstripped_text):]
#     return (rstripped_text, whitespace)

def response_to_status_and_string(response: MessageBatchIndividualResponse) -> tuple[int, str]:
    response_id: Final[str] = response.custom_id
    result: Final[MessageBatchResult] = response.result
    result_type: Final[MessageBatchResultType] = result.type
    if result_type == MESSAGE_BATCH_RESULT_TYPE_SUCCEEDED:
        message: Final[Message] = result.message
        stop_reason: Final[MessageStopReason] = message.stop_reason
        text: str = message_to_str(message)
        if stop_reason == MESSAGE_STOP_REASON_END_TURN or stop_reason == MESSAGE_STOP_REASON_STOP_SEQUENCE:
            return (EXIT_SUCCESS, text)
        elif stop_reason == MESSAGE_STOP_REASON_REFUSAL:
            write_stderr(f"Response {response_id}: refused by classifier")
            return (EXIT_REFUSAL, text)
        write_stderr(f"Response {response_id}: truncated: {stop_reason}")
        return (EXIT_TRUNCATED, text + DELIMITER_STOP_REASON_OR_ERROR + stop_reason)
    elif result_type == MESSAGE_BATCH_RESULT_TYPE_ERRORED:
        error: Final[AnthropicErrorObject] = result.error.error
        error_message: Final[str] = error.type + ": " + error.message
        write_stderr(f"Response {response_id}: error: {error_message}")
        return (EXIT_ERROR, DELIMITER_STOP_REASON_OR_ERROR + error_message)
    else:
        write_stderr(f"Response {response_id}: unhandled result type: {result_type}")
        return (EXIT_ERROR, DELIMITER_ERROR + "unhandled result type: " + result_type)

async def batch_completion(
        messages: Iterable[MessageParam],
        system_prompt: Optional[str] = None,
        client: Optional[AsyncAnthropic] = None,
        model_id: str = DEFAULT_MODEL_ID,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens_per_request: int = DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        n: int = 1,
        thinking: Optional[int | Literal["adaptive"]] = None,
        effort: Optional[str] = None,
        caching: Optional[str] = None,
) -> AsyncIterator[MessageBatchIndividualResponse]:

    if thinking is not None:
        if isinstance(thinking, int):
            if not (thinking > 0): raise ValueError("‘thinking’ must be a positive integer")
        elif thinking != "adaptive":
            raise ValueError("‘thinking’ must be either a positive integer or the string “adaptive”")
        if thinking == "adaptive":
            # TODO: implement adaptive thinking
            write_stderr("WARNING: adaptive thinking is not yet implemented")
            thinking = None
    if effort is not None:
        # TODO: implement effort
        write_stderr("WARNING: effort is not yet implemented")
        effort = None
    if caching is not None:
        # TODO: Implement caching
        write_stderr("WARNING: caching is not yet implemented")
        caching = None

    # TODO: more bounds/type checking

    if client is None:
        client = AsyncAnthropic()
    elif not isinstance(client, AsyncAnthropic):
        raise TypeError("‘client’ must be an instance of ‘AsyncAnthropic’")

    async for response in message_batch_submit_and_poll_until_completion(
            client=client,
            poll_interval=poll_interval,
            requests=(
                MessageBatchCreateRequest(
                    custom_id=str(k),
                    params=MessageCreateParamsNonStreaming(
                        model=model_id,
                        temperature=temperature,
                        max_tokens=max_output_tokens_per_request,
                        system=system_prompt,
                        messages=messages,
                        thinking=(
                            ThinkingConfigEnabledParam(type="enabled", budget_tokens=thinking)
                            if isinstance(thinking, int) else
                            ThinkingConfigDisabledParam(type="disabled")
                        ),
                    ),
                )
                for k in range(0, n)
            ),
    ):
        yield response

########################################################################
# COMMAND-LINE OPTIONS

def convert_thinking_argument(thinking: str) -> Optional[int | Literal["adaptive"]]:
    if not thinking: return None
    if thinking == "adaptive": return "adaptive"
    thinking_int: Final[int] = int(thinking)
    if not (thinking_int > 0): raise ValueError("‘thinking’ must be a positive integer")
    return thinking_int

argument_parser = ArgumentParser(
    description="Minimal plain-text interface for Anthropic's large language model \"Claude\", utilising batch mode for lower inference costs.",
)
argument_parser.add_argument("-m", "--model", help="Model ID (default: %(default)s)", type=str, dest="model_id", default=DEFAULT_MODEL_ID)
argument_parser.add_argument("-t", "--temperature", help="Sampling temperature from 0.0 to 1.0 (default: %(default)s)", type=float, dest="temperature", default=DEFAULT_TEMPERATURE)
argument_parser.add_argument("-n", "--number", help="Number of completion requests (default: %(default)s)", type=int, dest="n", metavar="OUTPUT_TOKENS", default=1)
argument_parser.add_argument("-i", "--interval", help="Polling interval in seconds (default: %(default)s)", type=int, dest="poll_interval", metavar="POLLING_INTERVAL", default=DEFAULT_POLL_INTERVAL)
argument_parser.add_argument("-l", "--length", help="Maximum number of output tokens per request (default: %(default)s)", type=int, dest="max_output_tokens_per_request", default=DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST)
argument_parser.add_argument("-c", "--caching", help="Cache control (5 minutes or 1 hour)", type=str, dest="caching", choices=("5m", "1h"), default=None)
argument_parser.add_argument("--no-caching", help="Disable caching", action="store_const", const=None, dest="caching")
argument_parser.add_argument("--thinking", help="Thinking (token budget or the keyword \"adaptive\")", nargs="?", default=None, const="adaptive", dest="thinking", type=convert_thinking_argument)
argument_parser.add_argument("--effort", help="Effort", type=str, dest="effort", default=None, choices=("low", "medium", "high", "max"))

########################################################################
# MAIN PROGRAM

async def main(argv: Iterable[str]) -> int:

    write_stderr("argv: " + repr(argv[1:]))
    args = argument_parser.parse_args(argv[1:])
    write_stderr("command-line options: " + repr(args))

    system_prompt: str = ""
    messages: Final[list[MessageParam]] = []
    input_text: str = ""
    input_prefix: str = ""
    input_messages: list[MessageParam] = []
    last_message_role: Optional[MessageRole] = None

    write_stderr("Reading conversation history from standard input...")
    input_text = stdin.read()

    write_stderr("Parsing input...")
    [input_prefix, input_messages] = message_split(input_text)
    write_stderr(f"Parsed {len(input_messages)} messages.")
    system_prompt += input_prefix
    messages += input_messages
    if not messages: raise ValueError("At least one message is required")

    last_message_role = messages[-1]["role"]
    while last_message_role == MESSAGE_ROLE_USER:
        write_stderr("Conversation ends with a ‘Human’ message; reading further input...")
        input_text = stdin.read()
        write_stderr("Parsing input...")
        [input_prefix, input_messages] = split_messages(input_text, last_message_role)
        write_stderr(f"Parsed {len(input_messages)} messages.")
        if input_prefix:
            # TODO: handle other content block types
            messages[-1]["content"][-1]["text"] += input_prefix
        messages += input_messages
        last_message_role = messages[-1]["role"]

    # Delete the last message if it's empty.
    if not "\n\n".join(map(content_to_str, messages[-1]["content"])).strip():
        write_stderr("Discarding empty ‘Assistant’ message at end of conversation.")
        del messages[-1]

    # DEBUGGING
    write_stderr(f"system_prompt = {system_prompt!r}\nmessages = {messages!r}")
    # return EXIT_SUCCESS

    client = AsyncAnthropic()

    exit_status: int = 0
    response_count: int = 0
    async for response in batch_completion(
            client=client,
            model_id=args.model_id,
            temperature=args.temperature,
            max_output_tokens_per_request=args.max_output_tokens_per_request,
            system_prompt=system_prompt,
            messages=messages,
            n=args.n,
            poll_interval=args.poll_interval,
            thinking=args.thinking,
            effort=args.effort,
            caching=args.caching,
    ):
        [status, response_string] = response_to_status_and_string(response)
        exit_status |= status
        if response_count: print(DELIMITER_COMPLETION, end="")
        write_stdout(response_string)
        response_count += 1
    write_stderr("Done.")
    return exit_status

if __name__ == "__main__":
    logging_config(level=LOGGING_LEVEL_DEBUG)
    exit(asyncio_run(main(argv)))
