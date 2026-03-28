# vim: set expandtab shiftwidth=4 softtabstop=4:

"""Qt tool: ChimeraLLM chat panel."""

import threading
import html

from Qt.QtCore import QObject, Qt, QThread, Signal
from Qt.QtGui import QTextCursor  # noqa: F401 – used by _append_html if needed later
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
        self.input_line.setText(text)
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
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._open_settings)
        top.addWidget(settings_btn)
        layout.addLayout(top)

        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setAcceptRichText(True)
        self.chat_view.setMinimumHeight(200)
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
        clear.triggered.connect(lambda: self._clear_chat())
        menu.addAction(clear)

    def _clear_chat(self):
        self._api_messages.clear()
        self.chat_view.clear()

    def _append_html(self, html_snippet: str):
        self.chat_view.append(html_snippet)
        sb = self.chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _fmt_user(self, text: str) -> str:
        esc = html.escape(text)
        return f'<p><b>You:</b><br/>{esc}</p>'

    def _fmt_assistant(self, text: str) -> str:
        esc = html.escape(text)
        return f'<p><b>Assistant:</b><br/>{esc}</p>'

    def _fmt_cmd(self, cmd: str, result: str) -> str:
        c = html.escape(cmd)
        r = html.escape(result[:4000])
        return f'<p><b>Command:</b> <code>{c}</code><br/><b>Result:</b> {r}</p>'

    def _fmt_note(self, text: str) -> str:
        return f'<p><i>{html.escape(text)}</i></p>'

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
        self._append_html(f'<p><b>Error:</b> {html.escape(err)}</p>')

    def _open_settings(self):
        dlg = QDialog(self.tool_window.ui_area)
        dlg.setWindowTitle("ChimeraLLM Settings")
        form = QFormLayout(dlg)

        # --- Copilot section ---
        cp_group = QGroupBox("GitHub Copilot")
        cp_layout = QFormLayout(cp_group)

        cp_check = QCheckBox("Use GitHub Copilot instead of API")
        cp_check.setChecked(bool(self._prefs.use_copilot))
        cp_layout.addRow(cp_check)

        cp_model_combo = QComboBox()
        cp_model_combo.setEditable(True)
        cp_model_combo.setMinimumWidth(300)
        saved_cp_model = self._prefs.copilot_model or "gpt-4o"
        # Start with saved model, fetch full list on demand
        cp_model_combo.addItem(saved_cp_model)
        cp_model_combo.setCurrentText(saved_cp_model)
        cp_layout.addRow("Model:", cp_model_combo)

        cp_refresh_btn = QPushButton("Refresh models")
        cp_layout.addRow(cp_refresh_btn)

        def _refresh_models():
            cp_model_combo.clear()
            models = agent_mod.fetch_copilot_models()
            cp_model_combo.addItems(models)
            if saved_cp_model in models:
                cp_model_combo.setCurrentText(saved_cp_model)

        cp_refresh_btn.clicked.connect(_refresh_models)

        # Auth status + login button
        from chimerallm.copilot_auth import get_token
        has_token = get_token() is not None
        cp_status = QLabel("Logged in" if has_token else "Not logged in")
        cp_status.setStyleSheet("color: green;" if has_token else "color: #a00;")
        cp_layout.addRow("Status:", cp_status)

        cp_login_btn = QPushButton("Login with GitHub")
        cp_layout.addRow(cp_login_btn)

        def _do_login():
            try:
                from chimerallm.copilot_auth import start_device_flow, poll_for_token
                flow = start_device_flow()
                code = flow["user_code"]
                url = flow["verification_uri"]

                # Show the code to the user
                from Qt.QtWidgets import QMessageBox
                msg_box = QMessageBox(dlg)
                msg_box.setWindowTitle("GitHub Login")
                msg_box.setText(
                    f"Open <b>{url}</b> in your browser and enter this code:\n\n"
                    f"<h2>{code}</h2>\n\n"
                    f"Waiting for authorization..."
                )
                msg_box.setStandardButtons(QMessageBox.StandardButton.Cancel)
                msg_box.setModal(False)
                msg_box.show()

                # Poll in a thread so UI stays responsive
                import threading
                result_holder = [None, None]  # [token, error]

                def _poll():
                    try:
                        token = poll_for_token(flow["device_code"], flow.get("interval", 5))
                        result_holder[0] = token
                    except Exception as e:
                        result_holder[1] = str(e)

                t = threading.Thread(target=_poll, daemon=True)
                t.start()

                # Use a timer to check completion
                from Qt.QtCore import QTimer
                timer = QTimer(dlg)

                def _check():
                    if t.is_alive():
                        return
                    timer.stop()
                    msg_box.close()
                    if result_holder[0]:
                        cp_status.setText("Logged in")
                        cp_status.setStyleSheet("color: green;")
                        self.session.logger.info("GitHub Copilot login successful.")
                    else:
                        cp_status.setText("Login failed")
                        cp_status.setStyleSheet("color: #a00;")
                        self.session.logger.error(f"Copilot login failed: {result_holder[1]}")

                timer.timeout.connect(_check)
                timer.start(500)

            except Exception as e:
                cp_status.setText(f"Error: {e}")
                cp_status.setStyleSheet("color: #a00;")

        cp_login_btn.clicked.connect(_do_login)

        form.addRow(cp_group)

        # --- API section ---
        api_group = QGroupBox("OpenAI-compatible API")
        api_layout = QFormLayout(api_group)

        url_edit = QLineEdit()
        url_edit.setText(self._prefs.api_base_url or "")
        url_edit.setPlaceholderText("https://openrouter.ai/api/v1")
        api_layout.addRow("API endpoint URL:", url_edit)

        key_edit = QLineEdit()
        key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        key_edit.setText(self._prefs.api_key or "")
        api_layout.addRow("API key:", key_edit)

        model_edit = QLineEdit()
        model_edit.setText(self._prefs.model or "gpt-4o")
        api_layout.addRow("Model:", model_edit)

        temp = QDoubleSpinBox()
        temp.setRange(0.0, 2.0)
        temp.setSingleStep(0.1)
        temp.setValue(float(self._prefs.temperature))
        api_layout.addRow("Temperature:", temp)

        form.addRow(api_group)

        # --- Shared settings ---
        iters = QSpinBox()
        iters.setRange(1, 50)
        iters.setValue(int(self._prefs.max_iterations))
        form.addRow("Max iterations:", iters)

        # Toggle API group enabled based on Copilot checkbox
        def _toggle_api_group(checked):
            api_group.setEnabled(not checked)

        cp_check.toggled.connect(_toggle_api_group)
        _toggle_api_group(cp_check.isChecked())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        form.addRow(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._prefs.use_copilot = cp_check.isChecked()
        self._prefs.copilot_model = cp_model_combo.currentText().strip() or "gpt-4o"
        self._prefs.api_base_url = url_edit.text().strip()
        self._prefs.model = model_edit.text().strip() or "gpt-4o"
        self._prefs.temperature = float(temp.value())
        self._prefs.max_iterations = int(iters.value())
        self._prefs.api_key = key_edit.text().strip()
        self._prefs.save()
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

            if self._tool._prefs.use_copilot:
                # Pre-fetch session info so the model doesn't waste a
                # round trip calling get_session_info.
                session_info = get_info()
                reply = agent_mod.run_agent_copilot(
                    self._tool.session,
                    self._tool._api_messages,
                    self._tool._prefs,
                    callbacks,
                    session_info=session_info,
                )
            else:
                reply = agent_mod.run_agent(
                    self._tool.session,
                    self._tool._api_messages,
                    self._tool._prefs,
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
