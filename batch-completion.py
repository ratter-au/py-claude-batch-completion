#!/usr/bin/env python3
"""Minimal plain-text interface for Anthropic's large language model
"Claude", utilising batch mode for lower inference costs.

The input and output are formatted to match the text in the model's
context window as closely as possible. The conversation history is read
from standard input. The user is expected to maintain it in a text file,
adding their own message at the end, before using this script to request
a response from Claude.

Responses are written to standard output. If more than one response is
requested, they will be separated by a "file separator" control
character (U+001C).

Complete messages (`stop_reason = "end_turn"`) will end with the "Human"
turn marker. Incomplete messages will have the `stop_reason` value
appended after an "end of text" character (U+0003). Requests which could
not be completed will be indicated with a "bell" character (U+0007)
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
from anthropic.types.message import Message
from anthropic.types.stop_reason import StopReason
from anthropic.types.message_param import MessageParam
from anthropic.types.text_block_param import TextBlockParam
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
# CONSTANTS

# Values for the `messages[].role` property in the Anthropic API.
ROLE_HUMAN: Final[Literal["user"]] = "user"
ROLE_ASSISTANT: Final[Literal["assistant"]] = "assistant"

# Values for the `messages[].content[].type` property in the Anthropic API.
CONTENT_TYPE_TEXT: Final[Literal["text"]] = "text"
CONTENT_TYPE_THINKING: Final[Literal["thinking"]] = "thinking"

DELIMITER_COMPLETION: Final[str] = "\x1C"
"""Delimiter character between output completions."""

DELIMITER_STOP_REASON: Final[str] = "\x03"
"""Delimiter character before `stop_reason`."""

DELIMITER_ERROR: Final[str] = "\x07"
"""Delimiter character before an error message."""

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

DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST: Final[int] = 4096
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

EXIT_SUCCESS = 0
"""Exit status indicating success."""

EXIT_ERROR = 1
"""Exit status indicating an error."""

EXIT_TRUNCATED = 2
"""Exit status indicating a truncated message."""

EXIT_REFUSAL = 4
"""Exit status indicating a refusal."""

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

def message_to_string(message: Message) -> str:
    """Convert a `Message` object to a string.

    Only `text` and `thinking` content blocks will be converted. Other
    content types will be replaced with an "unhandled content type"
    placeholder message.

    """
    return "".join(
        (
            content if isinstance(content, str) else
            content.text if content.type == CONTENT_TYPE_TEXT else
            (THINKING_BLOCK_START_TAG + content.thinking + THINKING_BLOCK_END_TAG) if content.type == CONTENT_TYPE_THINKING else
            f"[unhandled content type: {content.type}]"
        ) for content in message.content
    ) + (
        LABEL_HUMAN if message.stop_reason == "end_turn"
        else message.stop_sequence if message.stop_reason == "stop_sequence"
        else ""
    )

def split_messages(text: str) -> tuple[str, list[MessageParam]]:
    """Split text into a system prompt followed by strictly alternating ‘Human’ & ‘Assistant’ messages."""

    system_prompt: str = ""
    messages: Final[list[MessageParam]] = []
    remaining_text: str = text
    next_human_label: str = ""
    next_human_message_text: str = ""
    next_assistant_label: str = ""
    next_assistant_message_text: str = ""
    turn_index: int = 0

    [system_prompt, next_human_label, remaining_text] = remaining_text.partition(LABEL_HUMAN)
    while next_human_label:
        turn_index += 1
        [next_human_message_text, next_assistant_label, remaining_text] = remaining_text.partition(LABEL_ASSISTANT)
        next_human_message_text = next_human_message_text.rstrip()
        if not next_human_message_text: raise ValueError(f"Empty ‘Human’ message at turn {turn_index}")
        messages.append(MessageParam(
            role=ROLE_HUMAN,
            content=[
                TextBlockParam(
                    type=CONTENT_TYPE_TEXT,
                    text=next_human_message_text,
                ),
            ],
        ))
        if not next_assistant_label: break
        turn_index += 1
        [next_assistant_message_text, next_human_label, remaining_text] = remaining_text.partition(LABEL_HUMAN)
        next_assistant_message_text = next_assistant_message_text.rstrip()
        if not next_assistant_message_text: raise ValueError(f"Empty ‘Assistant’ message at turn {turn_index}")
        messages.append(MessageParam(
            role=ROLE_ASSISTANT,
            content=[
                TextBlockParam(
                    type=CONTENT_TYPE_TEXT,
                    text=next_assistant_message_text,
                ),
            ],
        ))
    return (system_prompt, messages)

async def submit_message_batch_and_poll_until_completion(client: AsyncAnthropic, poll_interval: int, requests: list[MessageBatchCreateRequest]) -> list[MessageBatchIndividualResponse]:
    write_stderr("Submitting message batch...")
    batch: MessageBatch = await client.messages.batches.create(requests=requests)
    batch_id: Final[str] = batch.id
    write_stderr(f"Submitted message batch ({batch_id})")
    while batch.processing_status != "ended":
        write_stderr(f"Sleeping for {poll_interval} seconds")
        await asyncio_sleep(poll_interval)
        write_stderr(f"Checking status of message batch ({batch_id})")
        batch = await client.messages.batches.retrieve(batch_id)
        write_stderr(batch.request_counts.model_dump_json())
    write_stderr(f"Message batch ({batch_id}) finished processing; fetching results...")
    responses: Final[list[MessageBatchIndividualResponse]] = []
    async for response in await client.messages.batches.results(batch_id):
        responses.append(response)
    write_stderr(f"Finished receiving results for message batch ({batch_id})")
    return responses

def response_to_status_and_string(response: MessageBatchIndividualResponse) -> tuple[int, str]:
    response_id: Final[str] = response.custom_id
    result: Final[MessageBatchResult] = response.result
    result_type: Final[str] = result.type
    if result_type == "succeeded":
        message: Final[Message] = result.message
        stop_reason: Final[str] = message.stop_reason
        text: str = LABEL_ASSISTANT + message_to_string(message) + (LABEL_HUMAN if stop_reason != "stop_sequence" else message.stop_sequence)
        if stop_reason == "end_turn" or stop_reason == "stop_sequence":
            return (EXIT_SUCCESS, text)
        if stop_reason == "refusal":
            write_stderr(f"response {response_id}: refused by classifier")
            return (EXIT_REFUSAL, text + DELIMITER_STOP_REASON + stop_reason)
        write_stderr(f"response {response_id}: truncated: {stop_reason}")
        return (EXIT_TRUNCATED, text + DELIMITER_STOP_REASON + stop_reason)
    if result_type == "errored":
        error: Final[AnthropicErrorObject] = result.error.error
        error_message: Final[str] = error.type + ": " + error.message
        write_stderr(f"response {response_id}: error: {error_message}")
        return (EXIT_ERROR, DELIMITER_ERROR + error_message)
    write_stderr(f"response {response_id}: unhandled result type: {result_type}")
    return (EXIT_ERROR, DELIMITER_ERROR + "unhandled result type: " + result_type)

########################################################################
# COMMAND-LINE OPTIONS

# argument_parser = ArgumentParser(
# )

# TODO

########################################################################
# MAIN PROGRAM

async def main() -> int:
    system_prompt: str = ""
    messages: list[MessageParam] = []
    last_message_role: Optional[str] = None
    last_message_stop_reason: Optional[str] = None

    write_stderr("Reading & parsing input...")
    [system_prompt, messages] = split_messages(stdin.read())
    write_stderr(f"Finished parsing input: {len(messages)} messages.")
    if not messages: raise ValueError("At least one message is required")
    last_message = messages[-1]
    # if last_message["role"] != ROLE_HUMAN: raise ValueError("Input must end with a ‘Human’ message")
    # if not message_to_text(last_message): raise ValueError("Last ‘Human’ message is empty")

    # TODO: accept command-line options for cache control
    # last_message["content"][-1]["cache_control"] = CACHE_CONTROL_EPHEMERAL_5M

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

    requests: Final[list[MessageBatchCreateRequest]] = [
        MessageBatchCreateRequest(
            custom_id="0",
            params=MessageCreateParamsNonStreaming(
                model=DEFAULT_MODEL_ID,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_OUTPUT_TOKENS_PER_REQUEST,
                system=system_prompt,
                messages=messages,
            ),
        ),
    ]

    responses: Final[list[MessageBatchIndividualResponse]] = await submit_message_batch_and_poll_until_completion(client, DEFAULT_POLL_INTERVAL, requests)

    exit_status: int = 0
    response_strings: Final[list[str]] = []
    write_stderr("Parsing responses...")
    for (status, response_string) in map(response_to_status_and_string, responses):
        exit_status |= status
        response_strings.append(response_string)
    write_stderr("Finished parsing responses; writing to standard output.")
    write_stdout(DELIMITER_COMPLETION.join(response_strings))
    return exit_status

exit(asyncio_run(main()))
