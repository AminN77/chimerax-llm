# vim: set expandtab shiftwidth=4 softtabstop=4:

"""Qt tool: ChimeraLLM chat panel."""

from __future__ import annotations

import html
import threading
import time
from typing import Optional

from Qt.QtCore import QObject, QRectF, Qt, QThread, Signal, QTimer
from Qt.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut, QTextCursor
from Qt.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPlainTextEdit,
    QLineEdit,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QTabWidget,
    QWidget,
    QMessageBox,
    QCheckBox,
)

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow

from chimerallm.settings import get_settings
from chimerallm import agent as agent_mod

# Dark-theme-safe bubble colors (Qt rich text subset)
_USER_BG = "#1e3a5c"
_USER_TITLE = "#e0e8f0"
_USER_BODY = "#d0d8e0"
_ASSIST_BG = "#2a2a3a"
_ASSIST_TITLE = "#e0e0e0"
_ASSIST_BODY = "#d8d8d8"
_TOOL_BG = "#1a2e1a"
_TOOL_TITLE = "#a8d4a8"
_TOOL_BODY = "#c0d0c0"
_NOTE_BG = "#2a2a2a"
_NOTE_BODY = "#a0a0a0"
_ERR_BG = "#3a1a1a"
_ERR_BODY = "#f0c0c0"


class _StatusSpinner(QWidget):
    """Small circular loading indicator (rotating arc)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0
        self.setFixedSize(18, 18)

    def advance(self):
        self._angle = (self._angle + 22) % 360
        self.update()

    def reset(self):
        self._angle = 0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx = self.width() * 0.5
        cy = self.height() * 0.5
        r = min(cx, cy) - 2.0
        p.translate(cx, cy)
        p.rotate(self._angle)
        pen = QPen(QColor(176, 186, 206))
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        rect = QRectF(-r, -r, 2.0 * r, 2.0 * r)
        # Gap at the trailing end (classic spinner)
        p.drawArc(rect, 45 * 16, 270 * 16)


class _ChimeraLLMQt(QObject):
    """Holds Qt signals; ToolInstance is not a QObject, so signals must live here (PyQt6)."""

    append_chat_html = Signal(str)
    command_request = Signal(str, object)
    session_info_request = Signal(object)
    agent_finished = Signal(str)
    agent_failed = Signal(str)
    streaming_start = Signal()
    streaming_delta = Signal(str)
    streaming_end = Signal()
    status_update = Signal(str)


class ChimeraLLMTool(ToolInstance):
    """Dockable chat UI; agent runs in a worker thread."""

    SESSION_ENDURING = False
    SESSION_SAVE = True
    help = "help:user/tools/chimerallm.html"

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self._api_messages: list = []
        self._api_messages_lock = threading.Lock()

        self._stream_start_pos: int = 0
        self._stream_buffer: str = ""
        self._suppress_next_assistant_finish: bool = False
        self._status_base: str = ""

        self._qt = _ChimeraLLMQt()
        self._qt.append_chat_html.connect(self._append_html, Qt.ConnectionType.QueuedConnection)
        self._qt.command_request.connect(self._on_command_request, Qt.ConnectionType.QueuedConnection)
        self._qt.session_info_request.connect(self._on_session_info_request, Qt.ConnectionType.QueuedConnection)
        self._qt.agent_finished.connect(self._on_agent_finished, Qt.ConnectionType.QueuedConnection)
        self._qt.agent_failed.connect(self._on_agent_failed, Qt.ConnectionType.QueuedConnection)
        self._qt.streaming_start.connect(self._on_streaming_start, Qt.ConnectionType.QueuedConnection)
        self._qt.streaming_delta.connect(self._on_streaming_delta, Qt.ConnectionType.QueuedConnection)
        self._qt.streaming_end.connect(self._on_streaming_end, Qt.ConnectionType.QueuedConnection)
        self._qt.status_update.connect(self._on_status_update, Qt.ConnectionType.QueuedConnection)

        # Not named _settings: ChimeraX ToolInstance may use _settings for the tool name (str).
        self._prefs = get_settings(session)
        self._agent_worker = None  # must retain QThread until finished (see _send_message)

        self.tool_window = MainToolWindow(self)
        self.tool_window.fill_context_menu = self._fill_context_menu
        self._build_ui()
        self.tool_window.manage("side")

    def submit_prompt(self, text: str):
        """Queue a user message (e.g. from the `chimerallm` command)."""
        text = (text or "").strip()
        if not text:
            return
        self.prompt_input.setPlainText(text)
        self._send_message()

    def _build_ui(self):
        tw = self.tool_window
        area = tw.ui_area
        layout = QVBoxLayout(area)
        layout.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        top.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Clear the transcript and LLM conversation context")
        clear_btn.clicked.connect(lambda: self._clear_chat())
        top.addWidget(clear_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setToolTip("Stop the current LLM request")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_agent)
        top.addWidget(self._cancel_btn)
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._open_settings)
        top.addWidget(settings_btn)
        layout.addLayout(top)

        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setAcceptRichText(True)
        self.chat_view.setMinimumHeight(200)
        layout.addWidget(self.chat_view, stretch=1)

        self._status_row = QWidget()
        self._status_row.setVisible(False)
        status_layout = QHBoxLayout(self._status_row)
        status_layout.setContentsMargins(0, 2, 0, 2)
        status_layout.setSpacing(8)
        self._status_spinner = _StatusSpinner(self._status_row)
        status_layout.addWidget(self._status_spinner, alignment=Qt.AlignmentFlag.AlignTop)
        self._status_text = QLabel("")
        self._status_text.setWordWrap(True)
        status_layout.addWidget(self._status_text, stretch=1)
        layout.addWidget(self._status_row)

        # QTimer parent must be a QObject; ToolInstance is not QObject.
        self._status_anim_timer = QTimer(self._qt)
        self._status_anim_timer.setInterval(45)
        self._status_anim_timer.timeout.connect(self._tick_status_animation)

        row = QHBoxLayout()
        self.prompt_input = QPlainTextEdit()
        self.prompt_input.setMinimumHeight(120)
        self.prompt_input.setPlaceholderText(
            "Describe what you want in ChimeraX…\n"
            "(Enter for a new line — Ctrl+Enter or the Send button to submit)"
        )
        self.prompt_input.setTabChangesFocus(False)
        row.addWidget(self.prompt_input, stretch=1)
        self._send_btn = QPushButton("Send")
        self._send_btn.setToolTip("Send prompt (same as Ctrl+Enter)")
        self._send_btn.clicked.connect(self._send_message)
        row.addWidget(self._send_btn, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(row)

        for seq in ("Ctrl+Return", "Ctrl+Enter"):
            sc = QShortcut(QKeySequence(seq), self.prompt_input)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(self._send_message)

        self._append_html(
            '<table width="100%" cellpadding="6" cellspacing="0" border="0">'
            '<tr><td bgcolor="' + _NOTE_BG + '"><font color="' + _NOTE_BODY + '">'
            "<i>Configure a provider in Settings. Commands run by the agent appear below.</i>"
            "</font></td></tr></table>"
        )

    def _session_log(self, level: str, msg: str) -> None:
        if not getattr(self._prefs, "log_to_chimerax", True):
            return
        fn = getattr(self.session.logger, level, None)
        if callable(fn):
            fn(msg)

    def _tick_status_animation(self):
        if not self._status_row.isVisible() or not self._status_base:
            return
        self._status_spinner.advance()

    def _on_status_update(self, message: str):
        if not message:
            self._status_anim_timer.stop()
            self._status_row.setVisible(False)
            self._status_base = ""
            return
        self._status_base = message.rstrip(".")
        self._status_text.setText(self._status_base)
        self._status_spinner.reset()
        self._status_row.setVisible(True)
        self._status_anim_timer.start()

    def _fill_context_menu(self, menu, x, y):
        from Qt.QtGui import QAction

        clear = QAction("Clear chat", menu)
        clear.triggered.connect(lambda: self._clear_chat())
        menu.addAction(clear)

    def _clear_chat(self):
        with self._api_messages_lock:
            self._api_messages.clear()
        self.chat_view.clear()

    def _append_html(self, html_snippet: str):
        self.chat_view.append(html_snippet)
        sb = self.chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    @staticmethod
    def _bubble_row_user(body: str) -> str:
        esc = html.escape(body).replace("\n", "<br/>")
        return (
            '<table width="100%" cellpadding="8" cellspacing="0" border="0">'
            '<tr><td width="15%"></td>'
            f'<td bgcolor="{_USER_BG}" align="left">'
            f'<font color="{_USER_TITLE}"><b>You</b></font><br/>'
            f'<font color="{_USER_BODY}">{esc}</font>'
            "</td></tr></table>"
        )

    @staticmethod
    def _bubble_row_assistant(body: str) -> str:
        esc = html.escape(body).replace("\n", "<br/>")
        return (
            '<table width="100%" cellpadding="8" cellspacing="0" border="0">'
            "<tr>"
            f'<td bgcolor="{_ASSIST_BG}" align="left">'
            f'<font color="{_ASSIST_TITLE}"><b>Assistant</b></font><br/>'
            f'<font color="{_ASSIST_BODY}">{esc}</font>'
            "</td>"
            '<td width="15%"></td>'
            "</tr></table>"
        )

    @staticmethod
    def _bubble_row_tool(cmd: str, result: str) -> str:
        c = html.escape(cmd)
        r = html.escape(result[:4000]).replace("\n", "<br/>")
        return (
            '<table width="100%" cellpadding="8" cellspacing="0" border="0">'
            "<tr>"
            f'<td bgcolor="{_TOOL_BG}" align="left">'
            f'<font color="{_TOOL_TITLE}"><b>ChimeraX command</b></font><br/>'
            f'<font color="{_TOOL_BODY}"><code>{c}</code><br/><b>Result:</b> {r}</font>'
            "</td>"
            '<td width="15%"></td>'
            "</tr></table>"
        )

    @staticmethod
    def _bubble_row_note(body: str) -> str:
        esc = html.escape(body).replace("\n", "<br/>")
        return (
            '<table width="100%" cellpadding="8" cellspacing="0" border="0">'
            "<tr>"
            f'<td bgcolor="{_NOTE_BG}" align="left">'
            f'<font color="{_NOTE_BODY}"><i>{esc}</i></font>'
            "</td>"
            '<td width="15%"></td>'
            "</tr></table>"
        )

    @staticmethod
    def _bubble_row_error(body: str) -> str:
        esc = html.escape(body).replace("\n", "<br/>")
        return (
            '<table width="100%" cellpadding="8" cellspacing="0" border="0">'
            "<tr>"
            f'<td bgcolor="{_ERR_BG}" align="left">'
            f'<font color="{_ERR_BODY}"><b>Error</b><br/>{esc}</font>'
            "</td>"
            '<td width="15%"></td>'
            "</tr></table>"
        )

    def _fmt_user(self, text: str) -> str:
        return self._bubble_row_user(text)

    def _fmt_assistant(self, text: str) -> str:
        return self._bubble_row_assistant(text)

    def _fmt_cmd(self, cmd: str, result: str) -> str:
        return self._bubble_row_tool(cmd, result)

    def _fmt_note(self, text: str) -> str:
        return self._bubble_row_note(text)

    def _on_streaming_start(self):
        cur = self.chat_view.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        self._stream_start_pos = cur.position()
        self._stream_buffer = ""

    def _on_streaming_delta(self, text: str):
        self._stream_buffer += text
        cur = self.chat_view.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.insertText(text)
        sb = self.chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_streaming_end(self):
        if not self._stream_buffer.strip():
            return
        cur = self.chat_view.textCursor()
        cur.setPosition(self._stream_start_pos)
        cur.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cur.removeSelectedText()
        cur.insertHtml(self._fmt_assistant(self._stream_buffer))
        self._suppress_next_assistant_finish = True
        sb = self.chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_command_request(self, cmd: str, callback):
        try:
            from chimerax.core.commands import run

            r = run(self.session, cmd)
            out = str(r) if r is not None else "OK"
        except Exception as e:
            out = f"Error: {e}"
        callback(out)

    def _on_session_info_request(self, callback):
        callback(agent_mod.gather_session_info(self.session))

    def _run_command(self, cmd: str, cancelled: Optional[threading.Event] = None) -> str:
        ev = threading.Event()
        result_holder: list = [None]

        def cb(val):
            result_holder[0] = val
            ev.set()

        self._qt.command_request.emit(cmd, cb)
        deadline = time.monotonic() + 30.0
        while not ev.is_set():
            if cancelled is not None and cancelled.is_set():
                return "(cancelled)"
            if time.monotonic() > deadline:
                return "(timeout waiting for ChimeraX command)"
            ev.wait(0.15)
        return result_holder[0] if result_holder[0] is not None else "(no result)"

    def _run_session_info(self, cancelled: Optional[threading.Event] = None) -> str:
        ev = threading.Event()
        result_holder: list = [None]

        def cb(val):
            result_holder[0] = val
            ev.set()

        self._qt.session_info_request.emit(cb)
        deadline = time.monotonic() + 10.0
        while not ev.is_set():
            if cancelled is not None and cancelled.is_set():
                return "(cancelled)"
            if time.monotonic() > deadline:
                return "(timeout waiting for session info)"
            ev.wait(0.15)
        return result_holder[0] if result_holder[0] is not None else "(no result)"

    def _send_message(self):
        text = self.prompt_input.toPlainText().strip()
        if not text:
            return
        if self._agent_worker is not None and self._agent_worker.isRunning():
            self._session_log("warning", "ChimeraLLM is still working on the previous message.")
            return
        self.prompt_input.clear()
        self._suppress_next_assistant_finish = False
        self._append_html(self._fmt_user(text))

        self._send_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._qt.status_update.emit("Thinking")

        self._agent_worker = _AgentWorker(self, text)
        self._agent_worker.finished.connect(self._on_agent_worker_thread_finished)
        self._agent_worker.start()

    def _cancel_agent(self):
        w = self._agent_worker
        if w is not None and w.isRunning():
            w.request_cancel()
            w.terminate()

    def _on_agent_worker_thread_finished(self):
        self._send_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._qt.status_update.emit("")
        w = self._agent_worker
        self._agent_worker = None
        if w is not None:
            w.deleteLater()

    def _on_agent_finished(self, reply: str):
        if self._suppress_next_assistant_finish:
            self._suppress_next_assistant_finish = False
            return
        if reply:
            self._append_html(self._fmt_assistant(reply))

    def _on_agent_failed(self, err: str):
        self._append_html(self._bubble_row_error(err))

    def _open_settings(self):
        dlg = QDialog(self.tool_window.ui_area)
        dlg.setWindowTitle("ChimeraLLM Settings")
        dlg.setMinimumWidth(420)

        outer = QVBoxLayout(dlg)

        tabs = QTabWidget()
        outer.addWidget(tabs)

        # --- Tab: OpenAI-compatible API ---
        api_page = QWidget()
        api_layout = QFormLayout(api_page)

        url_edit = QLineEdit()
        url_edit.setText(self._prefs.api_base_url or "")
        url_edit.setPlaceholderText("https://openrouter.ai/api/v1")
        api_layout.addRow("API endpoint URL:", url_edit)

        key_edit = QLineEdit()
        key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_edit.setText(self._prefs.api_key or "")
        api_layout.addRow("API key:", key_edit)

        api_model_combo = QComboBox()
        api_model_combo.setEditable(True)
        api_model_combo.setMinimumWidth(280)
        saved_api_model = self._prefs.model or "gpt-4o"
        api_model_combo.addItem(saved_api_model)
        api_model_combo.setCurrentText(saved_api_model)
        api_layout.addRow("Model:", api_model_combo)

        def _fill_api_models():
            keep = api_model_combo.currentText().strip() or saved_api_model
            if not key_edit.text().strip():
                self._session_log("warning", "Enter an API key to load models.")
                return
            try:
                models = agent_mod.fetch_openai_compatible_models(url_edit.text(), key_edit.text())
            except Exception as e:
                self._session_log("warning", f"Could not refresh API models: {e}")
                return
            api_model_combo.clear()
            api_model_combo.addItems(models)
            if keep in models:
                api_model_combo.setCurrentText(keep)
            else:
                api_model_combo.addItem(keep)
                api_model_combo.setCurrentText(keep)

        api_refresh_btn = QPushButton("Refresh models")
        api_refresh_btn.setToolTip("Fetch model IDs from the provider (OpenAI-compatible /v1/models)")
        api_refresh_btn.clicked.connect(_fill_api_models)
        api_layout.addRow(api_refresh_btn)

        temp = QDoubleSpinBox()
        temp.setRange(0.0, 2.0)
        temp.setSingleStep(0.1)
        temp.setValue(float(self._prefs.temperature))
        api_layout.addRow("Temperature:", temp)

        if self._prefs.api_key.strip():
            _fill_api_models()

        tabs.addTab(api_page, "OpenAI-compatible API")

        # --- Tab: GitHub Copilot ---
        cp_page = QWidget()
        cp_layout = QFormLayout(cp_page)

        from chimerallm.copilot_auth import get_copilot_token

        cp_status = QLabel()
        cp_model_combo = QComboBox()
        cp_model_combo.setEditable(True)
        cp_model_combo.setMinimumWidth(280)
        saved_cp_model = self._prefs.copilot_model or "gpt-4o"

        def _refresh_cp_status():
            has_token = get_copilot_token() is not None
            cp_status.setText("Signed in to GitHub" if has_token else "Not signed in")
            cp_status.setStyleSheet("color: #6a9e6a;" if has_token else "color: #c06060;")

        def _fill_copilot_models():
            keep = cp_model_combo.currentText().strip() or saved_cp_model
            cp_model_combo.clear()
            models = agent_mod.fetch_copilot_models()
            cp_model_combo.addItems(models)
            if keep in models:
                cp_model_combo.setCurrentText(keep)
            else:
                cp_model_combo.addItem(keep)
                cp_model_combo.setCurrentText(keep)

        _refresh_cp_status()
        _fill_copilot_models()

        cp_layout.addRow("Status:", cp_status)
        cp_login_btn = QPushButton("Sign in with GitHub…")
        cp_layout.addRow(cp_login_btn)

        cp_layout.addRow("Model:", cp_model_combo)

        cp_refresh_btn = QPushButton("Refresh models")
        cp_refresh_btn.setToolTip("Reload available Copilot model IDs from the registry")
        cp_refresh_btn.clicked.connect(_fill_copilot_models)
        cp_layout.addRow(cp_refresh_btn)

        cp_hint = QLabel("Sign in if needed, then use Refresh models to update the list.")
        cp_hint.setWordWrap(True)
        cp_layout.addRow(cp_hint)

        def _do_login():
            try:
                from chimerallm.copilot_auth import start_device_flow, poll_for_token

                flow = start_device_flow()
                code = flow["user_code"]
                url = flow["verification_uri"]

                msg_box = QMessageBox(dlg)
                msg_box.setWindowTitle("GitHub sign-in")
                msg_box.setText(
                    f"Open <b>{url}</b> in your browser and enter this code:\n\n"
                    f"<h2>{code}</h2>\n\n"
                    "Waiting for authorization…"
                )
                msg_box.setStandardButtons(QMessageBox.StandardButton.Cancel)
                msg_box.setModal(False)
                msg_box.show()

                result_holder = [None, None]

                def _poll():
                    try:
                        token = poll_for_token(flow["device_code"], flow.get("interval", 5))
                        result_holder[0] = token
                    except Exception as e:
                        result_holder[1] = str(e)

                t = threading.Thread(target=_poll, daemon=True)
                t.start()

                timer = QTimer(dlg)

                def _check():
                    if t.is_alive():
                        return
                    timer.stop()
                    msg_box.close()
                    if result_holder[0]:
                        _refresh_cp_status()
                        _fill_copilot_models()
                        self._session_log("info", "GitHub Copilot sign-in successful.")
                    else:
                        self._session_log("error", f"Copilot sign-in failed: {result_holder[1]}")

                timer.timeout.connect(_check)
                timer.start(500)

            except Exception as e:
                self._session_log("error", f"Copilot sign-in error: {e}")

        cp_login_btn.clicked.connect(_do_login)
        tabs.addTab(cp_page, "GitHub Copilot")

        # Active tab = current provider
        tabs.setCurrentIndex(1 if self._prefs.use_copilot else 0)

        # Shared
        shared = QFormLayout()
        log_cb = QCheckBox("Write ChimeraLLM messages to the ChimeraX log")
        log_cb.setChecked(bool(getattr(self._prefs, "log_to_chimerax", True)))
        log_cb.setToolTip(
            "When off, ChimeraLLM does not emit LLM request lines, settings notices, "
            "or other log lines to the ChimeraX log."
        )
        shared.addRow(log_cb)
        iters = QSpinBox()
        iters.setRange(1, 50)
        iters.setValue(int(self._prefs.max_iterations))
        shared.addRow("Max tool rounds per message:", iters)
        outer.addLayout(shared)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        outer.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        use_copilot = tabs.currentIndex() == 1
        self._prefs.use_copilot = use_copilot
        if use_copilot:
            self._prefs.copilot_model = cp_model_combo.currentText().strip() or "gpt-4o"
        else:
            self._prefs.api_base_url = url_edit.text().strip()
            self._prefs.api_key = key_edit.text().strip()
            self._prefs.model = api_model_combo.currentText().strip() or "gpt-4o"
            self._prefs.temperature = float(temp.value())
        self._prefs.max_iterations = int(iters.value())
        self._prefs.log_to_chimerax = log_cb.isChecked()
        self._prefs.save()
        self._session_log("info", "ChimeraLLM settings saved.")

    def delete(self):
        # Never block the GUI thread waiting on the worker: it must process
        # QueuedConnection callbacks from the worker, or deadlock occurs.
        w = getattr(self, "_agent_worker", None)
        if w is not None:
            if w.isRunning():
                w.request_cancel()
                w.terminate()
            w.deleteLater()
        self._agent_worker = None
        super().delete()

    def take_snapshot(self, session, flags):
        data = super().take_snapshot(session, flags)
        with self._api_messages_lock:
            data["chimerallm_api_messages"] = list(self._api_messages)
        return data

    def set_state_from_snapshot(self, session, data):
        super().set_state_from_snapshot(session, data)
        msgs = data.get("chimerallm_api_messages")
        if msgs is None:
            msgs = data.get("chimeragpt_api_messages")
        if msgs is not None:
            with self._api_messages_lock:
                self._api_messages = list(msgs)


class _AgentWorker(QThread):
    """Runs the LLM agent loop off the UI thread; notifies UI via QObject signals."""

    def __init__(self, tool: ChimeraLLMTool, user_text: str):
        super().__init__(None)
        self._tool = tool
        self._user_text = user_text
        self._cancelled = threading.Event()

    def request_cancel(self) -> None:
        self._cancelled.set()

    def run(self):
        working = None
        try:
            with self._tool._api_messages_lock:
                working = list(self._tool._api_messages)
            working.append({"role": "user", "content": self._user_text})

            def on_status(msg: str):
                self._tool._qt.status_update.emit(msg)

            def on_stream_start():
                self._tool._qt.streaming_start.emit()

            def on_stream_delta(t: str):
                self._tool._qt.streaming_delta.emit(t)

            def on_stream_end():
                self._tool._qt.streaming_end.emit()

            def exec_cmd(cmd: str) -> str:
                short = cmd[:80] + ("…" if len(cmd) > 80 else "")
                self._tool._qt.status_update.emit(f"Running: {short}")
                self._tool._qt.append_chat_html.emit(
                    self._tool._fmt_note(f"Running: {cmd[:500]}{'…' if len(cmd) > 500 else ''}")
                )
                res = self._tool._run_command(cmd, self._cancelled)
                self._tool._qt.append_chat_html.emit(self._tool._fmt_cmd(cmd, res))
                self._tool._qt.status_update.emit("Thinking")
                return res

            def get_info() -> str:
                return self._tool._run_session_info(self._cancelled)

            def log_ui(msg: str) -> None:
                self._tool._qt.append_chat_html.emit(self._tool._fmt_note(f"[agent] {msg}"))

            callbacks = agent_mod.AgentCallbacks(
                execute_chimerax_command=exec_cmd,
                get_session_info=get_info,
                log_message=log_ui,
                on_streaming_start=on_stream_start,
                on_streaming_delta=on_stream_delta,
                on_streaming_end=on_stream_end,
                on_status=on_status,
            )

            if self._tool._prefs.use_copilot:
                session_info = get_info()
                reply = agent_mod.run_agent_copilot(
                    self._tool.session,
                    working,
                    self._tool._prefs,
                    callbacks,
                    session_info=session_info,
                    cancelled=self._cancelled,
                )
            else:
                reply = agent_mod.run_agent(
                    self._tool.session,
                    working,
                    self._tool._prefs,
                    callbacks,
                    cancelled=self._cancelled,
                )
            with self._tool._api_messages_lock:
                self._tool._api_messages[:] = working
            self._tool._qt.agent_finished.emit(reply or "")
        except Exception as e:
            with self._tool._api_messages_lock:
                if working is not None:
                    if (
                        working
                        and working[-1].get("role") == "user"
                        and working[-1].get("content") == self._user_text
                    ):
                        working.pop()
                    self._tool._api_messages[:] = working
            self._tool._qt.agent_failed.emit(str(e))
