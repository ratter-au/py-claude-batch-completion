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

Tools are not supported by this script.

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
)
from asyncio import (
    run as asyncio_run,
    sleep as asyncio_sleep,
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

########################################################################
# EXPORTS

__all__ = (
    "batch_completion",
    "main",
)

########################################################################
# CONSTANTS

# Values for the `messages[].role` property in the Anthropic API.
ROLE_HUMAN: Final[Literal["user"]] = "user"
ROLE_ASSISTANT: Final[Literal["assistant"]] = "assistant"

# Key added to a `MessageParam` dictionary by `split_messages()` to
# denote the system prompt.
MESSAGE_SYSTEM_INDICATOR: Final[Literal["is_system_prompt"]] = "is_system_prompt"

# Values for the `messages[].content[].type` property in the Anthropic API.
CONTENT_TYPE_TEXT: Final[Literal["text"]] = "text"
CONTENT_TYPE_THINKING: Final[Literal["thinking"]] = "thinking"

DELIMITER_COMPLETION: Final[str] = "\x1C"
"""Delimiter character between output completions."""

DELIMITER_STOP_REASON_OR_ERROR: Final[str] = "\x03"
"""Delimiter character before `stop_reason` or an error message."""

# Message labels as seen by the language model.
LABEL_HUMAN: Final[str] = "\n\nHuman: "
LABEL_ASSISTANT: Final[str] = "\n\nAssistant: "

# Start & end tags for `thinking` blocks.
THINKING_BLOCK_START_TAG: Final[str] = "<antml:thinking>"
THINKING_BLOCK_END_TAG: Final[str] = "</antml:thinking>"

DEFAULT_MODEL_ID: Final[str] = "claude-haiku-4-5-20251001"
"""Default model ID."""

DEFAULT_TEMPERATURE: Final[float] = 1.0
"""Default temperature."""

DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST: Final[int] = 32768
"""Default maximum number of output tokens to generate for any single API request."""

DEFAULT_POLL_INTERVAL: Final[int] = 30
"""Default interval between polling the status of the message batch."""

CACHE_CONTROL_EPHEMERAL_5M = CacheControlEphemeralParam(
    type="ephemeral",
    ttl="5m",
)
"""Five-minute cache control marker."""

CACHE_CONTROL_EPHEMERAL_1H = CacheControlEphemeralParam(
    type="ephemeral",
    ttl="1h",
)
"""One-hour cache control marker."""

# Exit statuses
EXIT_SUCCESS: Final[int] = 0
EXIT_ERROR: Final[int] = 1
EXIT_TRUNCATED: Final[int] = 2
EXIT_REFUSAL: Final[int] = 4

# Stop reasons
STOP_REASON_END_TURN: Final[Literal["end_turn"]] = "end_turn"
STOP_REASON_STOP_SEQUENCE: Final[Literal["stop_sequence"]] = "stop_sequence"
STOP_REASON_REFUSAL: Final[Literal["refusal"]] = "refusal"

# Batch processing statuses
BATCH_STATUS_ENDED: Final[Literal["ended"]] = "ended"

# Batch result types
BATCH_RESULT_SUCCEEDED: Final[Literal["succeeded"]] = "succeeded"
BATCH_RESULT_ERRORED: Final[Literal["errored"]] = "errored"

########################################################################
# FUNCTIONS

def write_stdout(text: str) -> None:
    """Write `text` to standard output, followed by a newline, and flush the stream."""
    print(text, flush=True)

def write_stderr(text: str) -> None:
    """Write `text` to standard error, followed by a newline, and flush the stream."""
    print(text, file=stderr, flush=True)

def split_trailing_whitespace(text: str) -> tuple[str, str]:
    """Strip trailing whitespace from `text` (via `rstrip()`) and return `(rstripped_text, whitespace)`."""
    rstripped_text = text.rstrip()
    whitespace = text[len(rstripped_text):]
    return (rstripped_text, whitespace)

def content_param_to_string(content: (str | TextBlockParam | ThinkingBlockParam)) -> str:
    if isinstance(content, str): return str
    content_type: Final[str] = content["type"]
    if content_type == CONTENT_TYPE_TEXT: return content["text"]
    if content_type == CONTENT_TYPE_THINKING: return (THINKING_BLOCK_START_TAG + content["thinking"] + THINKING_BLOCK_END_TAG)
    return f"[unhandled content type: {content_type}]"

def content_to_string(content: (str | TextBlock | ThinkingBlock)) -> str:
    if isinstance(content, str): return str
    content_type: Final[str] = content.type
    if content_type == CONTENT_TYPE_TEXT: return content.text
    if content_type == CONTENT_TYPE_THINKING: return (THINKING_BLOCK_START_TAG + content.thinking + THINKING_BLOCK_END_TAG)
    return f"[unhandled content type: {content_type}]"

def message_to_string(message: Message) -> str:
    """Convert a `Message` object to a string.

    Only `text` and `thinking` content blocks will be converted. Other
    content types will be replaced with an "unhandled content type"
    placeholder message.

    """
    stop_reason: Final[str] = message.stop_reason
    return (
        "".join(map(content_to_string, message.content))
        + (
            LABEL_HUMAN if stop_reason == STOP_REASON_END_TURN
            else message.stop_sequence if stop_reason == STOP_REASON_STOP_SEQUENCE
            else ""
        )
    )

def split_messages(text: str, initial_role: Optional[Literal["user", "assistant"]] = None) -> tuple[str, list[MessageParam]]:
    """Split text into a prefix followed by strictly alternating ‘Human’ & ‘Assistant’ messages.

    TODO:
    - Parse ‘thinking’ blocks.
    - Accept an iterable as input and produce output as an iterator (via `yield`).

    """
    messages: Final[list[MessageParam]] = []
    initial_text: str = ""
    remaining_text: str = text
    next_label: str = ""
    current_message_text: str = ""
    current_message_is_empty: bool = False
    turn_index: int = 0

    current_role: str = ("system" if not initial_role else initial_role)
    next_label_sought: str = (LABEL_ASSISTANT if initial_role == ROLE_HUMAN else LABEL_HUMAN)

    [initial_text, next_label, remaining_text] = remaining_text.partition(next_label_sought)
    while next_label:
        if current_message_is_empty: raise ValueError(f"Empty message at turn {turn_index} is not the last message in the conversation")
        turn_index += 1
        current_role = (ROLE_ASSISTANT if current_role == ROLE_HUMAN else ROLE_HUMAN)
        next_label_sought = (LABEL_ASSISTANT if current_role == ROLE_HUMAN else LABEL_HUMAN)
        [current_message_text, next_label, remaining_text] = remaining_text.partition(next_label_sought)
        current_message_text = current_message_text.rstrip()
        current_message_is_empty = not current_message_text
        messages.append(MessageParam(
            role=current_role,
            content=[
                TextBlockParam(
                    type=CONTENT_TYPE_TEXT,
                    text=current_message_text,
                ),
            ],
        ))
    return (initial_text, messages)

async def submit_message_batch_and_poll_until_completion(
        client: AsyncAnthropic,
        poll_interval: int,
        requests: Iterable[MessageBatchCreateRequest],
) -> AsyncIterator[MessageBatchIndividualResponse]:
    write_stderr("Submitting message batch...")
    batch: MessageBatch = await client.messages.batches.create(requests=requests)
    batch_id: Final[str] = batch.id
    write_stderr(f"Submitted message batch ({batch_id})")
    while batch.processing_status != BATCH_STATUS_ENDED:
        write_stderr(f"Sleeping for {poll_interval} seconds")
        await asyncio_sleep(poll_interval)
        write_stderr(f"Checking status of message batch ({batch_id})")
        batch = await client.messages.batches.retrieve(batch_id)
        write_stderr(batch.request_counts.model_dump_json())
    write_stderr(f"Message batch ({batch_id}) finished processing; fetching results...")
    async for response in client.messages.batches.results(batch_id):
        write_stderr(f"Received result for request {response.custom_id!r}")
        yield response
    write_stderr(f"Finished receiving results for message batch ({batch_id})")

def response_to_status_and_string(response: MessageBatchIndividualResponse) -> tuple[int, str]:
    response_id: Final[str] = response.custom_id
    result: Final[MessageBatchResult] = response.result
    result_type: Final[str] = result.type
    if result_type == BATCH_RESULT_SUCCEEDED:
        message: Final[Message] = result.message
        stop_reason: Final[str] = message.stop_reason
        text: str = LABEL_ASSISTANT + message_to_string(message) + (LABEL_HUMAN if stop_reason != STOP_REASON_STOP_SEQUENCE else message.stop_sequence)
        if stop_reason == STOP_REASON_END_TURN or stop_reason == STOP_REASON_STOP_SEQUENCE:
            return (EXIT_SUCCESS, text)
        if stop_reason == STOP_REASON_REFUSAL:
            write_stderr(f"Response {response_id}: refused by classifier")
            return (EXIT_REFUSAL, text + DELIMITER_STOP_REASON_OR_ERROR + stop_reason)
        write_stderr(f"Response {response_id}: truncated: {stop_reason}")
        return (EXIT_TRUNCATED, text + DELIMITER_STOP_REASON_OR_ERROR + stop_reason)
    if result_type == BATCH_RESULT_ERRORED:
        error: Final[AnthropicErrorObject] = result.error.error
        error_message: Final[str] = error.type + ": " + error.message
        write_stderr(f"Response {response_id}: error: {error_message}")
        return (EXIT_ERROR, DELIMITER_STOP_REASON_OR_ERROR + error_message)
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
) -> AsyncIterator[MessageBatchIndividualResponse]:

    # TODO: bounds checking on arguments

    # TODO: accept options for:
    # - thinking
    # - effort
    # - caching

    if not client:
        client = AsyncAnthropic()

    async for response in submit_message_batch_and_poll_until_completion(
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
                    ),
                )
                for k in range(0, n)
            ),
    ):
        yield response

########################################################################
# COMMAND-LINE OPTIONS

# argument_parser = ArgumentParser(
# )

# TODO

########################################################################
# MAIN PROGRAM

async def main() -> int:

    system_prompt: str = ""
    messages: Final[list[MessageParam]] = []
    input_text: str = ""
    input_prefix: str = ""
    input_messages: list[MessageParam] = []
    last_message_role: Optional[str] = None

    write_stderr("Reading conversation history from standard input...")
    input_text = stdin.read()

    write_stderr("Parsing input...")
    [input_prefix, input_messages] = split_messages(input_text)
    write_stderr(f"Parsed {len(input_messages)} messages.")
    system_prompt += input_prefix
    messages += input_messages
    if not messages: raise ValueError("At least one message is required")

    last_message_role = messages[-1]["role"]
    while last_message_role == ROLE_HUMAN:
        write_stderr("Conversation ends with a ‘Human’ message; reading further input...")
        input_text = stdin.read()
        write_stderr("Parsing input...")
        [input_prefix, input_messages] = split_messages(input_text, last_message_role)
        write_stderr(f"Parsed {len(input_messages)} messages.")
        if input_prefix:
            # TODO: handle thinking blocks
            messages[-1]["content"][-1]["text"] += input_prefix
        messages += input_messages
        last_message_role = messages[-1]["role"]

    # Delete the last message if it's empty.
    if not "".join(map(content_param_to_string, messages[-1]["content"])).strip():
        write_stderr("Discarding empty ‘Assistant’ message at end of conversation.")
        del messages[-1]

    # TODO: accept an option for cache control
    # last_message["content"][-1]["cache_control"] = CACHE_CONTROL_EPHEMERAL_5M

    # DEBUGGING
    # write_stderr(f"system_prompt = {system_prompt!r}\nmessages = {messages!r}")
    # return EXIT_SUCCESS

    client = AsyncAnthropic()

    # TODO: accept command-line options for:
    # - model ID
    # - temperature
    # - maximum output tokens per completion
    # - thinking
    # - effort
    # - number of completions
    # - caching
    # - polling interval

    exit_status: int = 0
    response_count: int = 0
    async for response in batch_completion(
            client=client,
            model_id=DEFAULT_MODEL_ID,
            temperature=DEFAULT_TEMPERATURE,
            max_output_tokens_per_request=DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST,
            system_prompt=system_prompt,
            messages=messages,
            poll_interval=DEFAULT_POLL_INTERVAL,
    ):
        [status, response_string] = response_to_status_and_string(response)
        exit_status |= status
        if response_count: print(DELIMITER_COMPLETION, end="")
        write_stdout(response_string)
    write_stderr("Done.")
    return exit_status

if __name__ == "__main__":
    exit(asyncio_run(main()))
