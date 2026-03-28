# vim: set expandtab shiftwidth=4 softtabstop=4:

"""LLM tool-calling agent loop for ChimeraX (OpenAI-compatible API + GitHub Copilot)."""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, List, Optional

from chimerallm.settings import get_settings
from chimerallm.system_prompt import SYSTEM_PROMPT

# HTTP timeout for OpenAI SDK (seconds). Prevents indefinite hangs on bad networks.
OPENAI_CLIENT_TIMEOUT = 60.0


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
    try:
        if not getattr(get_settings(session), "log_to_chimerax", True):
            return
    except Exception:
        pass

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
        on_streaming_start: Optional[Callable[[], None]] = None,
        on_streaming_delta: Optional[Callable[[str], None]] = None,
        on_streaming_end: Optional[Callable[[], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.execute_chimerax_command = execute_chimerax_command
        self.get_session_info = get_session_info
        self.log_message = log_message
        self.on_assistant_delta = on_assistant_delta
        self.on_iteration = on_iteration
        self.on_streaming_start = on_streaming_start
        self.on_streaming_delta = on_streaming_delta
        self.on_streaming_end = on_streaming_end
        self.on_status = on_status


def _sync_api_messages(api_messages: List[Dict[str, Any]], messages_with_system: List[Dict[str, Any]]) -> None:
    api_messages[:] = messages_with_system[1:]


def _tool_calls_list_from_accumulator(acc: Dict[int, Dict[str, str]]) -> List[Dict[str, Any]]:
    """Build OpenAI-style tool_calls list from streaming accumulator (sorted by index)."""
    out: List[Dict[str, Any]] = []
    for idx in sorted(acc.keys()):
        t = acc[idx]
        out.append(
            {
                "id": t.get("id", ""),
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "arguments": t.get("arguments", "") or "{}",
                },
            }
        )
    return out


def _merge_tool_call_delta(acc: Dict[int, Dict[str, str]], delta_tc: Any) -> None:
    """Merge one streamed tool_calls delta fragment into *acc*."""
    i = getattr(delta_tc, "index", None)
    if i is None:
        return
    if i not in acc:
        acc[i] = {"id": "", "name": "", "arguments": ""}
    tid = getattr(delta_tc, "id", None)
    if tid:
        acc[i]["id"] = tid
    fn = getattr(delta_tc, "function", None)
    if fn is not None:
        name = getattr(fn, "name", None)
        if name:
            acc[i]["name"] = name
        args = getattr(fn, "arguments", None)
        if args:
            acc[i]["arguments"] += args


def _append_tool_results_from_calls(
    messages: List[Dict[str, Any]],
    assistant_content: Optional[str],
    tool_calls: List[Dict[str, Any]],
    callbacks: AgentCallbacks,
) -> None:
    """Append assistant message with tool_calls and execute tools (serializable dict format)."""
    assistant_msg: Dict[str, Any] = {"role": "assistant", "content": assistant_content}
    assistant_msg["tool_calls"] = tool_calls
    messages.append(assistant_msg)

    for tc in tool_calls:
        fn = tc.get("function") or {}
        fname = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            tid = tc.get("id", "")
            result = "Error: invalid JSON in tool arguments"
            messages.append({"role": "tool", "tool_call_id": tid, "content": result})
            continue

        tid = tc.get("id", "")
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

        messages.append({"role": "tool", "tool_call_id": tid, "content": result[:8000]})


def _stream_chat_completion(
    client: Any,
    create_kwargs: Dict[str, Any],
    *,
    callbacks: AgentCallbacks,
    cancelled: Optional[threading.Event],
) -> tuple[str, Optional[List[Dict[str, Any]]]]:
    """Run a streaming chat completion; return (assistant_text, tool_calls or None).

    If *tool_calls* is non-empty, *assistant_text* may still contain streamed prefix text.
    """
    create_kwargs = dict(create_kwargs)
    create_kwargs["stream"] = True

    if callbacks.on_status:
        callbacks.on_status("Thinking...")

    content_parts: List[str] = []
    tc_acc: Dict[int, Dict[str, str]] = {}
    started_streaming = False

    stream = client.chat.completions.create(**create_kwargs)
    for chunk in stream:
        if cancelled is not None and cancelled.is_set():
            raise RuntimeError("Cancelled by user.")
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue

        c = getattr(delta, "content", None)
        if c is not None:
            if not started_streaming:
                if callbacks.on_streaming_start:
                    callbacks.on_streaming_start()
                started_streaming = True
            if c and callbacks.on_streaming_delta:
                callbacks.on_streaming_delta(c)
            content_parts.append(c)

        for dtc in getattr(delta, "tool_calls", None) or []:
            _merge_tool_call_delta(tc_acc, dtc)

    if started_streaming and callbacks.on_streaming_end:
        callbacks.on_streaming_end()

    full_text = "".join(content_parts)
    tool_list = _tool_calls_list_from_accumulator(tc_acc) if tc_acc else []
    if tool_list:
        return full_text, tool_list
    return full_text, None


def _run_agent_loop(
    session,
    api_messages: List[Dict[str, Any]],
    messages_with_system: List[Dict[str, Any]],
    client: Any,
    model: str,
    callbacks: AgentCallbacks,
    max_iterations: int,
    *,
    via_copilot: bool,
    temperature: Optional[float] = None,
    cancelled: Optional[threading.Event] = None,
) -> str:
    """Shared tool-calling loop for API and Copilot backends."""
    final_text = ""

    for iteration in range(max_iterations):
        if cancelled is not None and cancelled.is_set():
            raise RuntimeError("Cancelled by user.")

        if callbacks.on_iteration:
            callbacks.on_iteration(iteration)

        _log_llm_request(
            session,
            model=model,
            via_copilot=via_copilot,
            this_call_chars=_messages_context_chars(messages_with_system),
        )

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages_with_system,
            "tools": TOOLS,
            "tool_choice": "auto",
        }
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if via_copilot:
            initiator = "user" if iteration == 0 else "agent"
            create_kwargs["extra_headers"] = {"x-initiator": initiator}

        assistant_text, tool_calls = _stream_chat_completion(
            client, create_kwargs, callbacks=callbacks, cancelled=cancelled
        )

        if not tool_calls:
            messages_with_system.append({"role": "assistant", "content": assistant_text})
            _sync_api_messages(api_messages, messages_with_system)
            return assistant_text or final_text or ""

        _append_tool_results_from_calls(
            messages_with_system,
            assistant_text or None,
            tool_calls,
            callbacks,
        )

    if cancelled is not None and cancelled.is_set():
        raise RuntimeError("Cancelled by user.")

    messages_with_system.append(
        {
            "role": "user",
            "content": "Stop calling tools. Briefly summarize what was done and what may still be needed.",
        }
    )
    _log_llm_request(
        session,
        model=model,
        via_copilot=via_copilot,
        this_call_chars=_messages_context_chars(messages_with_system),
    )

    final_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages_with_system,
    }
    if temperature is not None:
        final_kwargs["temperature"] = temperature
    if via_copilot:
        final_kwargs["extra_headers"] = {"x-initiator": "agent"}

    text, _tc = _stream_chat_completion(
        client, final_kwargs, callbacks=callbacks, cancelled=cancelled
    )
    messages_with_system.append({"role": "assistant", "content": text})
    final_text = text
    _sync_api_messages(api_messages, messages_with_system)
    return final_text


def run_agent(
    session,
    api_messages: List[Dict[str, Any]],
    settings,
    callbacks: AgentCallbacks,
    cancelled: Optional[threading.Event] = None,
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
    client = OpenAI(
        api_key=api_key.strip(),
        base_url=base_url,
        timeout=OPENAI_CLIENT_TIMEOUT,
    )
    model = getattr(settings, "model", "gpt-4o") or "gpt-4o"
    temperature = float(getattr(settings, "temperature", 0.2))
    max_iterations = int(getattr(settings, "max_iterations", 10))

    messages_with_system: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *api_messages,
    ]

    return _run_agent_loop(
        session,
        api_messages,
        messages_with_system,
        client,
        model,
        callbacks,
        max_iterations,
        via_copilot=False,
        temperature=temperature,
        cancelled=cancelled,
    )


# ---------------------------------------------------------------------------
# GitHub Copilot backend (direct API via OpenAI SDK)
# ---------------------------------------------------------------------------

_COPILOT_BASE_URL = "https://api.githubcopilot.com"
_COPILOT_HEADERS = {
    "User-Agent": "ChimeraLLM/0.1",
    "Openai-Intent": "conversation-edits",
    "Copilot-Integration-Id": "vscode-chat",
}

# Fallback list used when models.dev is unreachable.
_COPILOT_MODELS_FALLBACK: List[str] = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-5-mini",
    "claude-sonnet-4",
    "gemini-2.5-pro",
]


def fetch_openai_compatible_models(base_url: str, api_key: str) -> List[str]:
    """List model IDs from ``GET {base}/v1/models`` (OpenAI-compatible providers).

    Uses the same base URL convention as ``OpenAI(base_url=...)`` (path usually ends
    with ``/v1``). Raises ``RuntimeError`` on HTTP errors or invalid JSON.
    """
    import urllib.error
    import urllib.request

    key = (api_key or "").strip()
    if not key:
        raise RuntimeError("API key is required to list models.")

    base = (base_url or "").strip().rstrip("/")
    if not base:
        base = "https://openrouter.ai/api/v1"
    if not base.endswith("/v1"):
        base = base + "/v1"
    url = base + "/models"

    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "User-Agent": "ChimeraLLM/0.1",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Models list failed ({e.code}): {body}") from e

    models = data.get("data") or []
    ids: List[str] = []
    for m in models:
        if isinstance(m, dict):
            mid = m.get("id")
            if isinstance(mid, str) and mid:
                ids.append(mid)
    if not ids:
        raise RuntimeError("No models returned by the API.")
    return sorted(ids)


def fetch_copilot_models() -> List[str]:
    """Fetch available GitHub Copilot model IDs from the models.dev registry."""
    import urllib.request

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
    cancelled: Optional[threading.Event] = None,
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

    from chimerallm.copilot_auth import get_copilot_token

    token = get_copilot_token()
    if not token:
        raise RuntimeError(
            "No Copilot token found. Click 'Login with GitHub' in ChimeraLLM Settings."
        )

    client = OpenAI(
        api_key=token,
        base_url=_COPILOT_BASE_URL,
        default_headers=_COPILOT_HEADERS,
        timeout=OPENAI_CLIENT_TIMEOUT,
    )

    model = getattr(settings, "copilot_model", "gpt-4o") or "gpt-4o"
    max_iterations = int(getattr(settings, "max_iterations", 10))

    sys_content = SYSTEM_PROMPT
    if session_info:
        sys_content += "\n\n## Current session state (auto-provided)\n" + session_info

    messages_with_system: List[Dict[str, Any]] = [
        {"role": "system", "content": sys_content},
        *api_messages,
    ]

    return _run_agent_loop(
        session,
        api_messages,
        messages_with_system,
        client,
        model,
        callbacks,
        max_iterations,
        via_copilot=True,
        temperature=None,
        cancelled=cancelled,
    )
