# vim: set expandtab shiftwidth=4 softtabstop=4:

"""LLM tool-calling agent loop for ChimeraX (OpenAI-compatible API + GitHub Copilot)."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from chimerallm.system_prompt import SYSTEM_PROMPT


def _log_llm_request(
    session,
    *,
    model: str,
    via_copilot: bool,
    this_call_chars: int,
) -> None:
    """Log each outbound LLM request with model, route, and context size estimates.

    The agent runs on a worker thread; ChimeraX's logger/GUI is not thread-safe, so we
    marshal logging onto the UI thread via session.ui.thread_safe when in GUI mode.
    """
    route = "copilot" if via_copilot else "api"
    msg = (
        "ChimeraLLM LLM request: model=%s route=%s request_chars=%d "
        "(serialized messages including system prompt)"
        % (model, route, this_call_chars)
    )

    def _do_log():
        session.logger.info(msg)

    try:
        ui = getattr(session, "ui", None)
        if ui is not None and getattr(ui, "is_gui", False) and hasattr(ui, "thread_safe"):
            ui.thread_safe(_do_log)
        else:
            _do_log()
    except Exception:
        pass


def _messages_context_chars(messages: List[Dict[str, Any]]) -> int:
    """Serialized size of the message list (approximate context for API calls)."""
    return len(json.dumps(messages, default=str))


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_chimerax_command",
            "description": (
                "Run one or more ChimeraX command-line commands. "
                "Separate multiple commands with semicolons. "
                "Returns command output or an error string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Full ChimeraX command text (e.g. 'open 1abc' or 'color red #1').",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_info",
            "description": (
                "Summarize open models, selection, and basic session state. "
                "Call this before acting if the user's request depends on what is loaded."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_message",
            "description": "Show a short status message to the user in the ChimeraLLM panel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Brief message (one or two lines).",
                    }
                },
                "required": ["message"],
            },
        },
    },
]


def gather_session_info(session) -> str:
    """Build a text summary of session state (call from main thread)."""
    lines: List[str] = []
    try:
        models = list(session.models.list())
    except Exception as e:
        return f"(Could not list models: {e})"

    if not models:
        lines.append("No open models.")
    else:
        lines.append(f"Open models ({len(models)}):")
        for m in models:
            mid = getattr(m, "id", None)
            name = getattr(m, "name", "?")
            mtype = type(m).__name__
            lines.append(f"  - id={mid!s} name={name!r} type={mtype}")
    try:
        sel_empty = session.selection.empty()
        lines.append(f"Selection empty: {sel_empty}")
        if not sel_empty:
            sm = session.selection.models()
            lines.append(f"Selected models: {len(sm)}")
    except Exception as e:
        lines.append(f"(Selection info unavailable: {e})")
    return "\n".join(lines)


class AgentCallbacks:
    """Callbacks supplied by the UI."""

    def __init__(
        self,
        execute_chimerax_command: Callable[[str], str],
        get_session_info: Callable[[], str],
        log_message: Callable[[str], None],
        on_assistant_delta: Optional[Callable[[str], None]] = None,
        on_iteration: Optional[Callable[[int], None]] = None,
    ):
        self.execute_chimerax_command = execute_chimerax_command
        self.get_session_info = get_session_info
        self.log_message = log_message
        self.on_assistant_delta = on_assistant_delta
        self.on_iteration = on_iteration


def run_agent(
    session,
    api_messages: List[Dict[str, Any]],
    settings,
    callbacks: AgentCallbacks,
) -> str:
    """
    Run one assistant turn. `api_messages` must already end with the latest user message.
    On success, `api_messages` is replaced with the full transcript (no system message).
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "The 'openai' package is required. Reinstall the bundle or: pip install openai"
        ) from e

    api_key = getattr(settings, "api_key", "") or ""
    if not api_key.strip():
        raise RuntimeError("API key is not set. Open ChimeraLLM Settings and save your key.")

    base_url = (getattr(settings, "api_base_url", "") or "").strip()
    if not base_url:
        base_url = "https://openrouter.ai/api/v1"
    client = OpenAI(api_key=api_key.strip(), base_url=base_url)
    model = getattr(settings, "model", "gpt-4o") or "gpt-4o"
    temperature = float(getattr(settings, "temperature", 0.2))
    max_iterations = int(getattr(settings, "max_iterations", 10))

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *api_messages,
    ]

    final_text = ""

    for iteration in range(max_iterations):
        if callbacks.on_iteration:
            callbacks.on_iteration(iteration)

        _log_llm_request(
            session,
            model=model,
            via_copilot=False,
            this_call_chars=_messages_context_chars(messages),
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=temperature,
        )

        msg = response.choices[0].message

        if msg.content:
            final_text = msg.content
            if callbacks.on_assistant_delta:
                callbacks.on_assistant_delta(msg.content)

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            _sync_api_messages(api_messages, messages)
            return msg.content or final_text or ""

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.content}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in msg.tool_calls
        ]
        messages.append(assistant_msg)

        for tc in msg.tool_calls:
            fname = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                result = "Error: invalid JSON in tool arguments"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                continue

            try:
                if fname == "execute_chimerax_command":
                    cmd = args.get("command", "")
                    result = callbacks.execute_chimerax_command(cmd)
                elif fname == "get_session_info":
                    result = callbacks.get_session_info()
                elif fname == "log_message":
                    callbacks.log_message(args.get("message", ""))
                    result = "Message shown to user."
                else:
                    result = f"Unknown tool {fname}"
            except Exception as e:
                result = f"Error executing tool {fname}: {e}"

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result[:8000]}
            )

    messages.append(
        {
            "role": "user",
            "content": "Stop calling tools. Briefly summarize what was done and what may still be needed.",
        }
    )
    _log_llm_request(
        session,
        model=model,
        via_copilot=False,
        this_call_chars=_messages_context_chars(messages),
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    text = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": text})
    final_text = text
    _sync_api_messages(api_messages, messages)
    return final_text


def _sync_api_messages(api_messages: List[Dict[str, Any]], messages_with_system: List[Dict[str, Any]]) -> None:
    api_messages[:] = messages_with_system[1:]


# ---------------------------------------------------------------------------
# GitHub Copilot backend (direct API via OpenAI SDK)
# ---------------------------------------------------------------------------

_COPILOT_BASE_URL = "https://api.githubcopilot.com"
_COPILOT_HEADERS = {
    "User-Agent": "ChimeraLLM/0.1",
    "Openai-Intent": "conversation-edits",
}

# Fallback list used when models.dev is unreachable.
_COPILOT_MODELS_FALLBACK: List[str] = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-5-mini",
    "claude-sonnet-4",
    "gemini-2.5-pro",
]


def fetch_copilot_models() -> List[str]:
    """Fetch available GitHub Copilot model IDs from the models.dev registry."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            "https://models.dev/api.json",
            headers={"User-Agent": "ChimeraLLM/0.1"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        provider = data.get("github-copilot", {})
        models = provider.get("models", {})
        return sorted(models.keys()) if models else _COPILOT_MODELS_FALLBACK
    except Exception:
        return _COPILOT_MODELS_FALLBACK


def run_agent_copilot(
    session,
    api_messages: List[Dict[str, Any]],
    settings,
    callbacks: AgentCallbacks,
    session_info: str = "",
) -> str:
    """Run one assistant turn via the GitHub Copilot API with native tool calling.

    Uses the ``x-initiator`` header (same approach as opencode) so that only the
    first request in the agentic loop is billed as a premium Copilot request.
    Follow-up requests carrying tool results use ``x-initiator: agent`` and are
    not counted against the user's premium-request quota.

    *session_info* is pre-fetched session state injected into the system prompt
    so the model already knows what is loaded.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "The 'openai' package is required. Reinstall the bundle or: pip install openai"
        ) from e

    from chimerallm.copilot_auth import get_token

    token = get_token()
    if not token:
        raise RuntimeError(
            "No Copilot token found. Click 'Login with GitHub' in ChimeraLLM Settings."
        )

    client = OpenAI(
        api_key=token,
        base_url=_COPILOT_BASE_URL,
        default_headers=_COPILOT_HEADERS,
    )

    model = getattr(settings, "copilot_model", "gpt-4o") or "gpt-4o"
    max_iterations = int(getattr(settings, "max_iterations", 10))

    sys_content = SYSTEM_PROMPT
    if session_info:
        sys_content += "\n\n## Current session state (auto-provided)\n" + session_info

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": sys_content},
        *api_messages,
    ]

    final_text = ""

    for iteration in range(max_iterations):
        if callbacks.on_iteration:
            callbacks.on_iteration(iteration)

        # First iteration is user-initiated (billed); follow-ups carrying
        # tool results are agent-initiated (free under Copilot billing).
        initiator = "user" if iteration == 0 else "agent"

        _log_llm_request(
            session,
            model=model,
            via_copilot=True,
            this_call_chars=_messages_context_chars(messages),
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            extra_headers={"x-initiator": initiator},
        )

        msg = response.choices[0].message

        if msg.content:
            final_text = msg.content
            if callbacks.on_assistant_delta:
                callbacks.on_assistant_delta(msg.content)

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            _sync_api_messages(api_messages, messages)
            return msg.content or final_text or ""

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.content}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in msg.tool_calls
        ]
        messages.append(assistant_msg)

        for tc in msg.tool_calls:
            fname = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                result = "Error: invalid JSON in tool arguments"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                continue

            try:
                if fname == "execute_chimerax_command":
                    cmd = args.get("command", "")
                    result = callbacks.execute_chimerax_command(cmd)
                elif fname == "get_session_info":
                    result = callbacks.get_session_info()
                elif fname == "log_message":
                    callbacks.log_message(args.get("message", ""))
                    result = "Message shown to user."
                else:
                    result = f"Unknown tool {fname}"
            except Exception as e:
                result = f"Error executing tool {fname}: {e}"

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result[:8000]}
            )

    messages.append(
        {
            "role": "user",
            "content": "Stop calling tools. Briefly summarize what was done and what may still be needed.",
        }
    )
    _log_llm_request(
        session,
        model=model,
        via_copilot=True,
        this_call_chars=_messages_context_chars(messages),
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_headers={"x-initiator": "agent"},
    )
    text = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": text})
    final_text = text
    _sync_api_messages(api_messages, messages)
    return final_text
