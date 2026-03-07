# `ratter-au/py-claude-batch-completion`

Ultra-minimal plain-text command-line interface for Anthropic's large
language model "Claude", utilising batch mode for lower inference costs.

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

## Command-line options

- `-m`, `--model`: model ID
- `-t`, `--temperature`: temperature
- `-n`, `--number`: number of completions to request
- `-i`, `--interval`: polling interval in seconds
- `-l`, `--length`: maximum number of output tokens per request
- `-c`, `--caching`: prompt caching interval (`5m`/`1h`)
- `--no-caching`: disable prompt caching
- `--thinking`: thinking (token budget or the keyword `adaptive`)
- `--effort`: effort (`low`/`medium`/`high`/`max`)

## Exit status

The script's exit status will be the bitwise `OR` of the following
values for all messages:

- 0: success (`end_turn` or `stop_sequence`)
- 1: error
- 2: truncation (`max_tokens`, `pause_turn`, *etc.*)
- 4: refusal

## To do

- Tests!!!

## License

The contents of this repository are released under
[the Creative Commons Attribution 4.0 International (CC-BY-4.0) license](./LICENSE.txt).
