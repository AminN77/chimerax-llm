# vim: set expandtab shiftwidth=4 softtabstop=4:

"""LLM tool-calling agent loop for ChimeraX (OpenAI-compatible API + opencode)."""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Callable, Dict, List, Optional

from chimerallm.system_prompt import SYSTEM_PROMPT

# Extra instructions appended to the system prompt when using opencode (text-based tool calling).
_OPENCODE_TOOL_INSTRUCTIONS = """

## Tool-calling protocol
You do NOT have native function-calling. Instead, output tool calls using the exact XML tag format below.
Each tool call must be on its own line(s):

<tool_call>{"name": "TOOL_NAME", "arguments": {ARGS}}</tool_call>

Available tools:
1. execute_chimerax_command – Run one or more ChimeraX commands (separate with semicolons).
   Arguments: {"command": "the command string"}
2. get_session_info – Get a summary of open models, selection, and session state.
   Arguments: {}
3. log_message – Show a short status note to the user.
   Arguments: {"message": "text"}

Rules:
- You may include multiple <tool_call> tags in a single response.
- After you output tool calls, STOP. The results will be provided in the next message so you can continue.
- When you are done (no more tools needed), reply with your final text WITHOUT any <tool_call> tags.
"""

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
# opencode backend
# ---------------------------------------------------------------------------

# Lenient regex: catches </tool_call>, </minimax:tool_call>, or any </..tool_call..> variant.
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</[\w:]*tool_call>",
    re.DOTALL,
)
_KNOWN_TOOLS = {"execute_chimerax_command", "get_session_info", "log_message"}


def _find_opencode() -> str:
    """Locate the opencode binary, searching common paths if needed."""
    import shutil

    path = shutil.which("opencode")
    if path:
        return path
    # macOS GUI apps often lack Homebrew / nvm paths
    for candidate in (
        "/opt/homebrew/bin/opencode",
        "/usr/local/bin/opencode",
        os.path.expanduser("~/.local/bin/opencode"),
        os.path.expanduser("~/.nvm/versions/node/default/bin/opencode"),
    ):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find 'opencode'. Make sure it is installed "
        "(npm i -g opencode-ai) and on your PATH."
    )


def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool-call dicts from assistant text (lenient)."""
    calls: List[Dict[str, Any]] = []

    # 1) Try structured regex (<tool_call>...</tool_call> and mangled variants)
    for m in _TOOL_CALL_RE.finditer(text):
        raw = (m.group(1) or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "name" in obj:
                calls.append(obj)
                continue
        except json.JSONDecodeError:
            pass
        # Model may have added junk around the JSON — extract innermost braces
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                obj = json.loads(raw[brace_start : brace_end + 1])
                if isinstance(obj, dict) and "name" in obj:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass

    # 2) Fallback: find bare {"name": "known_tool", ...} JSON in the text
    #    (handles models that mangle or omit delimiters entirely)
    if not calls:
        for m in re.finditer(r'\{\s*"name"\s*:', text):
            start = m.start()
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            try:
                obj = json.loads(text[start:end])
                if isinstance(obj, dict) and obj.get("name") in _KNOWN_TOOLS:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass

    return calls


def _strip_tool_calls(text: str) -> str:
    """Return text with tool-call blocks removed."""
    cleaned = _TOOL_CALL_RE.sub("", text)
    # Also strip bare tool-call JSON that the fallback parser would match
    cleaned = re.sub(
        r'\{\s*"name"\s*:\s*"(?:execute_chimerax_command|get_session_info|log_message)"[^}]*(?:\{[^}]*\}[^}]*)?\}',
        "", cleaned,
    )
    return cleaned.strip()


def _call_opencode(prompt: str, model: str, session_id: Optional[str] = None) -> tuple:
    """Call ``opencode run`` and return (response_text, session_id)."""
    oc = _find_opencode()
    cmd = [oc, "run", "--model", model, "--format", "json"]
    if session_id:
        cmd += ["--session", session_id]
    cmd.append(prompt)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    text_parts: List[str] = []
    sid = session_id
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if sid is None:
            sid = event.get("sessionID")
        etype = event.get("type")
        if etype == "text":
            part = event.get("part", {})
            text_parts.append(part.get("text", ""))

    return "".join(text_parts), sid


def fetch_opencode_models() -> List[str]:
    """Return the list of model IDs from ``opencode models``."""
    try:
        oc = _find_opencode()
        result = subprocess.run(
            [oc, "models"],
            capture_output=True, text=True, timeout=15,
        )
        models = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        return models if models else []
    except Exception:
        return []


def run_agent_opencode(
    session,
    api_messages: List[Dict[str, Any]],
    settings,
    callbacks: AgentCallbacks,
) -> str:
    """Run one assistant turn via the opencode CLI."""
    model = getattr(settings, "opencode_model", "github-copilot/claude-sonnet-4") or "github-copilot/claude-sonnet-4"
    max_iterations = int(getattr(settings, "max_iterations", 10))

    system = SYSTEM_PROMPT + _OPENCODE_TOOL_INSTRUCTIONS

    # Build the initial prompt: system context + full conversation history
    parts: List[str] = [system, ""]
    for msg in api_messages:
        role = msg["role"].upper()
        parts.append(f"[{role}]: {msg['content']}")
    initial_prompt = "\n".join(parts)

    session_id: Optional[str] = None
    final_text = ""
    prompt = initial_prompt

    for iteration in range(max_iterations):
        if callbacks.on_iteration:
            callbacks.on_iteration(iteration)

        response_text, session_id = _call_opencode(prompt, model, session_id)

        tool_calls = _parse_tool_calls(response_text)
        clean_text = _strip_tool_calls(response_text)

        if clean_text and callbacks.on_assistant_delta:
            callbacks.on_assistant_delta(clean_text)

        if not tool_calls:
            # No tool calls → final answer
            final_text = clean_text or response_text
            api_messages.append({"role": "assistant", "content": final_text})
            return final_text

        # Execute each tool call
        result_parts: List[str] = []
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            try:
                if name == "execute_chimerax_command":
                    cmd = args.get("command", "")
                    result = callbacks.execute_chimerax_command(cmd)
                elif name == "get_session_info":
                    result = callbacks.get_session_info()
                elif name == "log_message":
                    callbacks.log_message(args.get("message", ""))
                    result = "Message shown to user."
                else:
                    result = f"Unknown tool: {name}"
            except Exception as e:
                result = f"Error executing {name}: {e}"
            result_parts.append(f"[Tool {name}] {result[:8000]}")

        # Send tool results back via session continuation
        prompt = "Tool results:\n" + "\n".join(result_parts) + "\n\nContinue. Use more tool calls if needed, or give your final answer."

    # Max iterations exceeded – ask for summary
    summary_prompt = "Stop calling tools. Briefly summarize what was done and what may still be needed."
    response_text, _ = _call_opencode(summary_prompt, model, session_id)
    final_text = _strip_tool_calls(response_text) or response_text
    api_messages.append({"role": "assistant", "content": final_text})
    return final_text
