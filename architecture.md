# ChimeraLLM Architecture

ChimeraLLM is a ChimeraX bundle that lets users control UCSF ChimeraX through natural language by routing prompts through an LLM agent that translates them into ChimeraX commands.

## High-level flow

```
User prompt (natural language)
        |
        v
  +------------------+
  | ChimeraLLMTool   |  Qt chat UI (tool.py)
  | (main thread)    |
  +--------+---------+
           |
           v
  +------------------+
  | _AgentWorker     |  QThread — keeps UI responsive
  +--------+---------+
           |
           +--- use_opencode=False ---> run_agent()       [OpenAI-compatible API]
           |                             Uses native function-calling (tools param)
           |
           +--- use_opencode=True  ---> run_agent_opencode()  [opencode CLI]
                                         Uses text-based <tool_call> protocol
           |
           v
  +------------------+
  | LLM responds     |  Either native tool_calls or <tool_call> XML in text
  +--------+---------+
           |
           v
  +------------------+
  | Execute tools    |  Callbacks marshal to main thread via Qt signals
  | - execute_cmd    |  Runs ChimeraX commands (chimerax.core.commands.run)
  | - get_session    |  Gathers open models / selection state
  | - log_message    |  Shows notes in the chat panel
  +--------+---------+
           |
           v
  Feed results back to LLM, loop until done or max_iterations reached
           |
           v
  Final text reply displayed in chat
```

## File map

| File | Role |
|---|---|
| `bundle_info.xml` | ChimeraX bundle metadata: name, version, dependencies (`openai>=1.0`), registered tool + command |
| `src/__init__.py` | Bundle API entry point — wires up the tool, command, and class registration |
| `src/cmd.py` | Registers the `chimerallm` CLI command; opens the tool window and optionally queues a prompt |
| `src/tool.py` | Qt-based chat UI (`ChimeraLLMTool`), settings dialog, and `_AgentWorker` thread |
| `src/agent.py` | Agent loop logic for both backends, tool definitions, opencode subprocess helpers, tool-call parser |
| `src/settings.py` | Persistent settings via ChimeraX `Settings` class (API key, model, endpoint, opencode toggle, etc.) |
| `src/system_prompt.py` | System prompt: role definition + ChimeraX command reference sent to the LLM |

## Two backends

### 1. OpenAI-compatible API (`run_agent`)

- Uses the `openai` Python SDK with a configurable `base_url` (defaults to OpenRouter)
- Sends the system prompt + conversation history with native `tools` parameter
- LLM returns structured `tool_calls` — parsed and executed in a loop
- Supports temperature, model selection, and API key configuration

### 2. opencode CLI (`run_agent_opencode`)

- Invokes `opencode run --model <model> --format json` via subprocess
- Uses session IDs (`--session`) for multi-turn conversation continuity
- Since opencode doesn't support native function-calling, the system prompt is augmented with `_OPENCODE_TOOL_INSTRUCTIONS` that teach the LLM a text-based `<tool_call>` protocol
- `_parse_tool_calls()` extracts tool calls with a lenient 2-tier parser:
  1. Regex match on `<tool_call>...</tool_call>` (tolerates mangled closing tags like `</minimax:tool_call>`)
  2. Fallback: finds bare `{"name": "known_tool", ...}` JSON blobs via brace-depth matching
- Models are fetched dynamically via `opencode models` for the settings dropdown

## Threading model

ChimeraX's UI runs on the main thread. The LLM agent loop (which blocks on network I/O) runs in a `QThread` (`_AgentWorker`). Communication between the two uses Qt signals with `QueuedConnection`:

- **Worker -> Main**: `append_chat_html`, `command_request`, `session_info_request`, `agent_finished`, `agent_failed`
- **Main -> Worker**: Callbacks pass results back via `threading.Event` wait/set pairs

This ensures ChimeraX commands (`chimerax.core.commands.run`) always execute on the main thread, which ChimeraX requires.

## Settings

Stored via ChimeraX's `Settings` framework (auto-persisted to disk):

| Setting | Default | Notes |
|---|---|---|
| `api_key` | `""` | Explicit save (not auto-saved on change) |
| `model` | `"gpt-4o"` | For API backend |
| `temperature` | `0.2` | For API backend |
| `api_base_url` | `""` | Empty = OpenRouter |
| `use_opencode` | `False` | Toggle between API and opencode backends |
| `opencode_model` | `"github-copilot/claude-sonnet-4"` | For opencode backend |
| `max_iterations` | `10` | Shared — max tool-calling rounds per user message |
