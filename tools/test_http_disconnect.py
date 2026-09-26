#!/usr/bin/env python3
"""Exercise HTTP disconnect handling on loopback with no pipeline or robot."""

from http.client import HTTPConnection, RemoteDisconnected
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controller.motion_control_app import LocalControlHTTPServer
from src.controller.native_app import GeminiNativeRequestHandler


def windows_abort():
    # Use Windows' actual OSError -> ConnectionAbortedError mapping when here.
    if os.name == "nt":
        error = OSError(0, "Software caused connection abort", None, 10053)
        assert isinstance(error, ConnectionAbortedError)
        return error
    return ConnectionAbortedError("Software caused connection abort")


class HTTPDisconnectTests(unittest.TestCase):
    def setUp(self):
        self.manager = SimpleNamespace(
            _append_log=Mock(),
            status=Mock(return_value={"ok": True, "phase": "stopped"}),
            update_config=Mock(return_value={"ok": True}),
        )

    def handler(self):
        handler = object.__new__(GeminiNativeRequestHandler)
        handler.server = SimpleNamespace(manager=self.manager)
        handler._send_error_json = Mock()
        handler._send_json = Mock()
        return handler

    def test_only_disconnected_clients_are_quiet(self):
        server = object.__new__(LocalControlHTTPServer)
        server.manager = self.manager
        for error in (windows_abort(), ConnectionResetError(), BrokenPipeError()):
            with self.subTest(error=type(error).__name__):
                try:
                    raise error
                except OSError:
                    server.handle_error(None, ("127.0.0.1", 49977))
        self.manager._append_log.assert_not_called()
        for error in (RuntimeError("handler bug"), PermissionError("disk denied"), ConnectionRefusedError("backend refused")):
            self.manager._append_log.reset_mock()
            try:
                raise error
            except Exception:
                server.handle_error(None, ("127.0.0.1", 49977))
            args, kwargs = self.manager._append_log.call_args
            self.assertIn(str(error), args[0])
            self.assertEqual(kwargs, {"component": "http", "level": "error"})

    def test_aborted_config_body_is_not_reported_as_a_save_failure(self):
        handler = self.handler()
        handler._read_json = Mock(side_effect=windows_abort())
        with self.assertRaises(ConnectionAbortedError):
            handler._handle_config_update()
        self.manager.update_config.assert_not_called()
        handler._send_error_json.assert_not_called()
        handler._send_json.assert_not_called()

    def test_actual_config_write_failure_is_still_reported(self):
        handler = self.handler()
        handler._read_json = Mock(return_value={"human_height": 1.8})
        self.manager.update_config.side_effect = PermissionError("disk denied")
        handler._handle_config_update()
        handler._send_error_json.assert_called_once_with(500, "Could not save configuration: disk denied")

    def test_cancelled_download_uses_disconnect_path(self):
        handler = self.handler()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.wfile = io.BytesIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.log"
            path.write_text("test log", encoding="utf-8")
            self.manager.downloadable_log = Mock(return_value=path)
            with patch("src.controller.motion_control_app.shutil.copyfileobj", side_effect=windows_abort()):
                with self.assertRaises(ConnectionAbortedError):
                    handler._download_log()
        self.manager._append_log.assert_not_called()

    def test_aborted_response_does_not_stop_following_http_requests(self):
        handled = threading.Event()

        class ObservedServer(LocalControlHTTPServer):
            def handle_error(self, request, client_address):
                try:
                    super().handle_error(request, client_address)
                finally:
                    handled.set()

        class RequestHandler(GeminiNativeRequestHandler):
            def do_GET(self):
                if self.path == "/test-abort":
                    # Inject the same error at response write. The real
                    # ThreadingHTTPServer must close this accepted connection.
                    with patch.object(self.wfile, "write", side_effect=windows_abort()):
                        self._send_json(200, {"ok": True})
                    return
                if self.path == "/test-bug":
                    raise RuntimeError("handler bug")
                super().do_GET()

        with ObservedServer(("127.0.0.1", 0), RequestHandler, self.manager) as server:
            worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
            worker.start()
            try:
                for failure, expected_logs in (("/test-abort", 0), ("/test-bug", 1)):
                    handled.clear()
                    connection = HTTPConnection(*server.server_address, timeout=2)
                    try:
                        connection.request("GET", failure)
                        with self.assertRaises(RemoteDisconnected):
                            connection.getresponse()
                    finally:
                        connection.close()
                    self.assertTrue(handled.wait(2), "request exception did not reach server")
                    self.assertEqual(self.manager._append_log.call_count, expected_logs)
                    # No task is launched; only a fake status manager is used.
                    connection = HTTPConnection(*server.server_address, timeout=2)
                    try:
                        connection.request("GET", "/api/status")
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(json.loads(response.read()), {"ok": True, "phase": "stopped"})
                    finally:
                        connection.close()
            finally:
                server.shutdown()
                worker.join(timeout=2)
            self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main(verbosity=2)
