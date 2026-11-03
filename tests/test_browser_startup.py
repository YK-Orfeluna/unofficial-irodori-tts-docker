from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest


def test_browser_opens_only_after_own_server_starts(monkeypatch):
    import gradio_app

    server = SimpleNamespace(started=False)
    stopped = threading.Event()
    opened = threading.Event()
    urls = []

    def open_browser(url):
        urls.append(url)
        opened.set()

    monkeypatch.setattr(gradio_app.webbrowser, "open", open_browser)
    thread = threading.Thread(target=gradio_app._open_browser_when_ready, args=("http://127.0.0.1:7860", server, stopped))
    thread.start()
    try:
        assert not opened.wait(0.3)
        server.started = True
        assert opened.wait(5)
    finally:
        stopped.set()
        thread.join(timeout=5)
    assert urls == ["http://127.0.0.1:7860"]


def test_browser_wait_stops_when_server_exits(monkeypatch):
    import gradio_app

    stopped = threading.Event()

    opened = []
    monkeypatch.setattr(gradio_app.webbrowser, "open", opened.append)
    thread = threading.Thread(target=gradio_app._open_browser_when_ready, args=("http://127.0.0.1:7860", SimpleNamespace(started=False), stopped))
    thread.start()
    stopped.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not opened


def test_failed_startup_returns_error_without_opening_browser(monkeypatch):
    import gradio_app
    from irodori_tts_api import app as api

    monkeypatch.setattr("sys.argv", ["gradio_app.py", "--open-browser"])
    monkeypatch.setattr(gradio_app, "build_ui", lambda: SimpleNamespace(queue=lambda **kwargs: None))
    monkeypatch.setattr(gradio_app.gr, "mount_gradio_app", lambda app, *args, **kwargs: app)
    monkeypatch.setattr(api, "attach_api", lambda *args, **kwargs: None)
    monkeypatch.setattr(gradio_app, "_default_model_device", lambda: "cpu")
    monkeypatch.setattr(gradio_app, "_default_codec_device", lambda: "cpu")
    monkeypatch.setenv("IRODORI_MODEL_PRECISION", "fp32")
    monkeypatch.setenv("IRODORI_CODEC_PRECISION", "fp32")
    monkeypatch.setattr(gradio_app.uvicorn, "Server", lambda config: SimpleNamespace(started=False, run=lambda: None))
    monkeypatch.setattr(gradio_app.webbrowser, "open", lambda url: pytest.fail("browser opened"))
    with pytest.raises(SystemExit) as error:
        gradio_app.main()
    assert error.value.code == 3
