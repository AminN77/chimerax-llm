# vim: set expandtab shiftwidth=4 softtabstop=4:

"""Qt tool: ChimeraLLM chat panel."""

import threading
import html

from Qt.QtCore import QObject, Qt, QThread, Signal
from Qt.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QToolButton,
    QCheckBox,
    QComboBox,
    QGroupBox,
)

from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow

from chimerallm.settings import get_settings
from chimerallm import agent as agent_mod


class _ChimeraLLMQt(QObject):
    """Holds Qt signals; ToolInstance is not a QObject, so signals must live here (PyQt6)."""

    append_chat_html = Signal(str)
    command_request = Signal(str, object)
    session_info_request = Signal(object)
    agent_finished = Signal(str)
    agent_failed = Signal(str)


class ChimeraLLMTool(ToolInstance):
    """Dockable chat UI; agent runs in a worker thread."""

    SESSION_ENDURING = False
    SESSION_SAVE = True
    help = "help:user/tools/chimerallm.html"

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self._api_messages: list = []

        self._qt = _ChimeraLLMQt()
        self._qt.append_chat_html.connect(self._append_html, Qt.ConnectionType.QueuedConnection)
        self._qt.command_request.connect(self._on_command_request, Qt.ConnectionType.QueuedConnection)
        self._qt.session_info_request.connect(self._on_session_info_request, Qt.ConnectionType.QueuedConnection)
        self._qt.agent_finished.connect(self._on_agent_finished, Qt.ConnectionType.QueuedConnection)
        self._qt.agent_failed.connect(self._on_agent_failed, Qt.ConnectionType.QueuedConnection)

        self._settings = get_settings(session)
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
        self.input_line.setText(text)
        self._send_message()

    def _build_ui(self):
        tw = self.tool_window
        area = tw.ui_area
        layout = QVBoxLayout(area)
        layout.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        title = QLabel("Natural language → ChimeraX commands (LLM)")
        top.addWidget(title)
        top.addStretch()
        settings_btn = QToolButton()
        settings_btn.setText("Settings")
        settings_btn.clicked.connect(self._open_settings)
        top.addWidget(settings_btn)
        layout.addLayout(top)

        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setAcceptRichText(True)
        self.chat_view.setMinimumHeight(280)
        layout.addWidget(self.chat_view, stretch=1)

        row = QHBoxLayout()
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Describe what you want in ChimeraX…")
        self.input_line.returnPressed.connect(self._send_message)
        row.addWidget(self.input_line, stretch=1)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._send_message)
        row.addWidget(send_btn)
        layout.addLayout(row)

        self._append_html(
            "<p><i>Enter an API key in Settings. Commands run by the agent appear below.</i></p>"
        )

    def _fill_context_menu(self, menu, x, y):
        from Qt.QtGui import QAction

        clear = QAction("Clear chat", menu)
        clear.triggered.connect(self._clear_chat)
        menu.addAction(clear)

    def _clear_chat(self):
        self.chat_view.clear()
        self._api_messages.clear()

    def _append_html(self, html_snippet: str):
        self.chat_view.append(html_snippet)
        sb = self.chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _fmt_user(self, text: str) -> str:
        esc = html.escape(text)
        return f'<p style="background:#e8f4fc;padding:6px;border-radius:6px;"><b>You:</b><br/>{esc}</p>'

    def _fmt_assistant(self, text: str) -> str:
        esc = html.escape(text)
        return f'<p style="background:#f0f0f0;padding:6px;border-radius:6px;"><b>Assistant:</b><br/>{esc}</p>'

    def _fmt_cmd(self, cmd: str, result: str) -> str:
        c = html.escape(cmd)
        r = html.escape(result[:4000])
        return (
            f'<p style="font-family:monospace;font-size:11px;background:#fff8e6;padding:4px;">'
            f"<b>Command:</b> <code>{c}</code><br/><b>Result:</b> {r}</p>"
        )

    def _fmt_note(self, text: str) -> str:
        return f'<p style="color:#555;"><i>{html.escape(text)}</i></p>'

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

    def _run_command(self, cmd: str) -> str:
        ev = threading.Event()
        result_holder: list = [None]

        def cb(val):
            result_holder[0] = val
            ev.set()

        self._qt.command_request.emit(cmd, cb)
        ev.wait(timeout=600.0)
        return result_holder[0] if result_holder[0] is not None else "(no result)"

    def _run_session_info(self) -> str:
        ev = threading.Event()
        result_holder: list = [None]

        def cb(val):
            result_holder[0] = val
            ev.set()

        self._qt.session_info_request.emit(cb)
        ev.wait(timeout=60.0)
        return result_holder[0] if result_holder[0] is not None else "(no result)"

    def _send_message(self):
        text = self.input_line.text().strip()
        if not text:
            return
        if self._agent_worker is not None and self._agent_worker.isRunning():
            self.session.logger.warning("ChimeraLLM is still working on the previous message.")
            return
        self.input_line.clear()
        self._append_html(self._fmt_user(text))

        self._agent_worker = _AgentWorker(self, text)
        self._agent_worker.finished.connect(self._on_agent_worker_thread_finished)
        self._agent_worker.start()

    def _on_agent_worker_thread_finished(self):
        w = self._agent_worker
        self._agent_worker = None
        if w is not None:
            w.deleteLater()

    def _on_agent_finished(self, reply: str):
        self._append_html(self._fmt_assistant(reply))

    def _on_agent_failed(self, err: str):
        self._append_html(f'<p style="color:#a00;"><b>Error:</b> {html.escape(err)}</p>')

    def _open_settings(self):
        dlg = QDialog(self.tool_window.ui_area)
        dlg.setWindowTitle("ChimeraLLM Settings")
        form = QFormLayout(dlg)

        # --- opencode section ---
        oc_group = QGroupBox("opencode (GitHub Copilot / local models)")
        oc_layout = QFormLayout(oc_group)

        oc_check = QCheckBox("Use opencode instead of API")
        oc_check.setChecked(bool(self._settings.use_opencode))
        oc_layout.addRow(oc_check)

        oc_model_combo = QComboBox()
        oc_model_combo.setEditable(True)
        oc_model_combo.setMinimumWidth(300)
        # Populate with models from opencode CLI
        saved_oc_model = self._settings.opencode_model or "github-copilot/claude-sonnet-4"
        oc_model_combo.addItem(saved_oc_model)
        oc_model_combo.setCurrentText(saved_oc_model)
        oc_layout.addRow("opencode model:", oc_model_combo)

        oc_refresh_btn = QPushButton("Refresh models")
        oc_layout.addRow(oc_refresh_btn)

        def _refresh_models():
            oc_model_combo.clear()
            models = agent_mod.fetch_opencode_models()
            if models:
                oc_model_combo.addItems(models)
                if saved_oc_model in models:
                    oc_model_combo.setCurrentText(saved_oc_model)
            else:
                oc_model_combo.addItem("(could not fetch models)")

        oc_refresh_btn.clicked.connect(_refresh_models)

        form.addRow(oc_group)

        # --- API section ---
        api_group = QGroupBox("OpenAI-compatible API")
        api_layout = QFormLayout(api_group)

        url_edit = QLineEdit()
        url_edit.setText(self._settings.api_base_url or "")
        url_edit.setPlaceholderText("https://openrouter.ai/api/v1")
        api_layout.addRow("API endpoint URL:", url_edit)

        key_edit = QLineEdit()
        key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_edit.setText(self._settings.api_key or "")
        api_layout.addRow("API key:", key_edit)

        model_edit = QLineEdit()
        model_edit.setText(self._settings.model or "gpt-4o")
        api_layout.addRow("Model:", model_edit)

        temp = QDoubleSpinBox()
        temp.setRange(0.0, 2.0)
        temp.setSingleStep(0.1)
        temp.setValue(float(self._settings.temperature))
        api_layout.addRow("Temperature:", temp)

        form.addRow(api_group)

        # --- Shared settings ---
        iters = QSpinBox()
        iters.setRange(1, 50)
        iters.setValue(int(self._settings.max_iterations))
        form.addRow("Max iterations:", iters)

        # Toggle API group enabled based on opencode checkbox
        def _toggle_api_group(checked):
            api_group.setEnabled(not checked)

        oc_check.toggled.connect(_toggle_api_group)
        _toggle_api_group(oc_check.isChecked())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._settings.use_opencode = oc_check.isChecked()
        self._settings.opencode_model = oc_model_combo.currentText().strip() or "github-copilot/claude-sonnet-4"
        self._settings.api_base_url = url_edit.text().strip()
        self._settings.model = model_edit.text().strip() or "gpt-4o"
        self._settings.temperature = float(temp.value())
        self._settings.max_iterations = int(iters.value())
        self._settings.api_key = key_edit.text().strip()
        self._settings.save()
        self.session.logger.info("ChimeraLLM settings saved.")

    def delete(self):
        w = getattr(self, "_agent_worker", None)
        if w is not None:
            if w.isRunning():
                self.session.logger.status("Waiting for ChimeraLLM to finish…")
                if not w.wait(120000):
                    w.terminate()
                    w.wait(3000)
            w.deleteLater()
        self._agent_worker = None
        super().delete()

    def take_snapshot(self, session, flags):
        data = super().take_snapshot(session, flags)
        data["chimerallm_api_messages"] = self._api_messages
        return data

    def set_state_from_snapshot(self, session, data):
        super().set_state_from_snapshot(session, data)
        msgs = data.get("chimerallm_api_messages")
        if msgs is None:
            msgs = data.get("chimeragpt_api_messages")
        if msgs is not None:
            self._api_messages = msgs


class _AgentWorker(QThread):
    """Runs the LLM agent loop off the UI thread; notifies UI via QObject signals."""

    def __init__(self, tool: ChimeraLLMTool, user_text: str):
        super().__init__(None)
        self._tool = tool
        self._user_text = user_text

    def run(self):
        try:
            self._tool._api_messages.append({"role": "user", "content": self._user_text})

            def exec_cmd(cmd: str) -> str:
                self._tool._qt.append_chat_html.emit(
                    self._tool._fmt_note(f"Running: {cmd[:500]}{'…' if len(cmd) > 500 else ''}")
                )
                res = self._tool._run_command(cmd)
                self._tool._qt.append_chat_html.emit(self._tool._fmt_cmd(cmd, res))
                return res

            def get_info() -> str:
                return self._tool._run_session_info()

            def log_ui(msg: str) -> None:
                self._tool._qt.append_chat_html.emit(self._tool._fmt_note(f"[agent] {msg}"))

            callbacks = agent_mod.AgentCallbacks(
                execute_chimerax_command=exec_cmd,
                get_session_info=get_info,
                log_message=log_ui,
            )

            if self._tool._settings.use_opencode:
                reply = agent_mod.run_agent_opencode(
                    self._tool.session,
                    self._tool._api_messages,
                    self._tool._settings,
                    callbacks,
                )
            else:
                reply = agent_mod.run_agent(
                    self._tool.session,
                    self._tool._api_messages,
                    self._tool._settings,
                    callbacks,
                )
            self._tool._qt.agent_finished.emit(reply or "")
        except Exception as e:
            self._tool._qt.agent_failed.emit(str(e))
            if (
                self._tool._api_messages
                and self._tool._api_messages[-1].get("role") == "user"
                and self._tool._api_messages[-1].get("content") == self._user_text
            ):
                self._tool._api_messages.pop()
