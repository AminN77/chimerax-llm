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
           +--- use_copilot=False ---> run_agent()          [OpenAI-compatible API]
           |
           +--- use_copilot=True  ---> run_agent_copilot()  [GitHub Copilot API]
           |
           v  (both use native OpenAI-style tool calling)
  +------------------+
  | LLM responds     |  With tool_calls in structured format
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
| `src/agent.py` | Agent loop logic for both backends, tool definitions, Copilot model list |
| `src/copilot_auth.py` | GitHub Copilot OAuth authentication: token reading, device-flow login, token storage |
| `src/settings.py` | Persistent settings via ChimeraX `Settings` class (API key, model, endpoint, Copilot toggle, etc.) |
| `src/system_prompt.py` | Comprehensive system prompt: role definition + full ChimeraX command reference (~640 lines) |

## Two backends

### 1. OpenAI-compatible API (`run_agent`)

- Uses the `openai` Python SDK with a configurable `base_url` (defaults to OpenRouter)
- Sends the system prompt + conversation history with native `tools` parameter
- LLM returns structured `tool_calls` — parsed and executed in a loop
- Supports temperature, model selection, and API key configuration

### 2. GitHub Copilot (`run_agent_copilot`)

- Uses the `openai` Python SDK pointed at `https://api.githubcopilot.com`
- Same native tool-calling protocol as the API backend
- Authenticates with a GitHub OAuth token obtained via device flow
- Token is read from `~/.local/share/opencode/auth.json` (shared with opencode) or obtained via the built-in "Login with GitHub" button
- Session info is pre-fetched and injected into the system prompt so the model already knows what is loaded (avoids a wasted `get_session_info` round trip)

#### Copilot billing: `x-initiator` header

Copilot bills per premium request, not per token. To keep the agentic loop without burning quota, the backend uses the `x-initiator` header (same approach as [opencode](https://github.com/sst/opencode)):

- **First request** in each turn: `x-initiator: user` — billed as one premium request
- **Follow-up requests** (carrying tool results back to the model): `x-initiator: agent` — not counted against the premium quota

This means a single user prompt costs exactly **one premium request** regardless of how many tool-calling rounds the agent takes.

Default HTTP headers for Copilot also include `Copilot-Integration-Id: vscode-chat` (required by the Copilot gateway).

### Copilot authentication flow

```
1. User clicks "Login with GitHub" in Settings
2. Plugin calls GitHub device code endpoint (POST github.com/login/device/code)
   -> Returns: verification_uri, user_code, device_code
3. User opens URL in browser and enters code
4. Plugin polls github.com/login/oauth/access_token until authorized
5. OAuth access token exchanged for a Copilot JWT via GET api.github.com/copilot_internal/v2/token
6. OAuth token + Copilot JWT + expiry saved to ~/.local/share/opencode/auth.json
7. Before each API session, JWT is refreshed automatically when near expiry (using the stored OAuth token)
8. Copilot JWT used as Bearer token for api.githubcopilot.com requests
```

## Threading model

ChimeraX's UI runs on the main thread. The LLM agent loop (which blocks on network I/O) runs in a `QThread` (`_AgentWorker`). Communication between the two uses Qt signals with `QueuedConnection`:

- **Worker -> Main**: `append_chat_html`, `command_request`, `session_info_request`, `agent_finished`, `agent_failed`
- **Main -> Worker**: Callbacks pass results back via `threading.Event` wait/set pairs (with short polling so cancellation and timeouts are responsive)

This ensures ChimeraX commands (`chimerax.core.commands.run`) always execute on the main thread, which ChimeraX requires.

**Important:** The tool's `delete()` method must **never** call `QThread.wait()` on the GUI thread while the worker might be blocked waiting for a queued callback — that deadlocks ChimeraX. On close, the worker is cancelled and `terminate()` is used if still running.

The conversation transcript `_api_messages` is protected by a `threading.Lock` and updated atomically after each assistant turn.

OpenAI HTTP calls use a bounded timeout (`OPENAI_CLIENT_TIMEOUT` in `agent.py`) so a stuck network cannot hang the worker forever.

## Settings

Stored via ChimeraX's `Settings` framework (auto-persisted to disk):

| Setting | Default | Notes |
|---|---|---|
| `api_key` | `""` | Explicit save (not auto-saved on change) |
| `model` | `"gpt-4o"` | For API backend |
| `temperature` | `0.2` | For API backend |
| `api_base_url` | `""` | Empty = OpenRouter |
| `use_copilot` | `False` | Toggle between API and Copilot backends |
| `copilot_model` | `"gpt-4o"` | For Copilot backend |
| `max_iterations` | `10` | Shared — max tool-calling rounds per user message |
