"""Windows desktop application window launcher for LOKI.

Launches the local UI server and opens a dedicated, chrome-free desktop application window:
- Priority 1: Microsoft Edge Application Mode (--app=http://127.0.0.1:<port>)
- Priority 2: Google Chrome Application Mode
- Priority 3: Default desktop browser fallback
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.request
import webbrowser

import uvicorn

from ..agent import LokiAgent
from ..core.logging import get_logger
from ..utils.paths import get_data_dir
from .server import create_ui_app

logger = get_logger("ui.launcher")

# Common browser executable paths on Windows
EDGE_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

_local_appdata = os.environ.get("LOCALAPPDATA", "")
if _local_appdata:
    EDGE_PATHS.append(os.path.join(_local_appdata, r"Microsoft\Edge\Application\msedge.exe"))
    CHROME_PATHS.append(os.path.join(_local_appdata, r"Google\Chrome\Application\chrome.exe"))


def find_free_port(preferred_port: int = 11435) -> int:
    """Find an available port on 127.0.0.1, defaulting to preferred_port if free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred_port))
            return preferred_port
        except OSError:
            pass

    # Find arbitrary free ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_app_browser() -> tuple[str | None, str]:
    """Identify available browser binary that supports --app mode on Windows."""
    for p in EDGE_PATHS:
        if os.path.exists(p):
            return p, "edge"

    for p in CHROME_PATHS:
        if os.path.exists(p):
            return p, "chrome"

    return None, "system_default"


def launch_desktop_window(
    url: str,
    width: int = 1240,
    height: int = 820,
    profile_dir: Path | None = None,
) -> subprocess.Popen[Any] | None:
    """Open the local UI in a standalone desktop application window."""
    browser_exe, browser_type = _find_app_browser()

    if browser_exe and browser_type in ("edge", "chrome"):
        logger.info("Opening LOKI in desktop application window via %s", browser_type)
        if profile_dir is None:
            profile_dir = get_data_dir() / "browser_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)

        args = [
            browser_exe,
            f"--app={url}",
            f"--window-size={width},{height}",
            "--new-window",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        try:
            return subprocess.Popen(args)
        except OSError as e:
            logger.warning("Failed to launch %s in app mode: %s", browser_type, e)

    # Fallback to system default browser
    logger.info("Opening LOKI in default desktop browser: %s", url)
    webbrowser.open(url)
    return None


def wait_for_server_ready(
    url: str,
    timeout: float = 5.0,
    poll_interval: float = 0.1,
    server_thread: threading.Thread | None = None,
) -> bool:
    """Poll the UI server status endpoint until it responds with HTTP 200 or timeout expires.

    Args:
        url: Base URL of the local server (e.g. http://127.0.0.1:11435).
        timeout: Maximum seconds to wait before timing out.
        poll_interval: Seconds to wait between polling attempts.
        server_thread: Optional server thread to detect premature crashes.

    Returns:
        True if the server is responsive, False on timeout or early crash.
    """
    deadline = time.time() + timeout
    status_url = f"{url.rstrip('/')}/api/status"

    while time.time() < deadline:
        if server_thread is not None and not server_thread.is_alive():
            logger.error("LOKI UI server thread terminated prematurely during startup.")
            return False

        try:
            req = urllib.request.Request(
                status_url,
                headers={"User-Agent": "LokiHealthCheck/1.0"},
            )
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass

        time.sleep(poll_interval)

    return False


def _probe_loki_server(url: str, timeout: float = 0.3) -> bool:
    """Check if a healthy LOKI UI server is responsive at the specified URL."""
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/api/status",
            headers={"User-Agent": "LokiInstanceCheck/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _get_server_info_path() -> Path:
    """Return path to the active server metadata file."""
    return get_data_dir() / "server.json"


def _read_active_server_info() -> dict[str, Any] | None:
    """Read active server metadata if the recorded instance is still alive."""
    info_path = _get_server_info_path()
    if not info_path.exists():
        return None
    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
        url = data.get("url")
        if url and _probe_loki_server(url):
            return data
        info_path.unlink(missing_ok=True)
    except Exception:
        try:
            info_path.unlink(missing_ok=True)
        except OSError:
            pass
    return None


def _write_server_info(url: str, port: int) -> None:
    """Persist active server metadata for single-instance discovery."""
    try:
        info_path = _get_server_info_path()
        info_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"url": url, "port": port, "pid": os.getpid()}
        info_path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as e:
        logger.debug("Could not write server info file: %s", e)


def _clear_server_info() -> None:
    """Remove server metadata file if it belongs to this process."""
    try:
        info_path = _get_server_info_path()
        if info_path.exists():
            data = json.loads(info_path.read_text(encoding="utf-8"))
            if data.get("pid") == os.getpid():
                info_path.unlink(missing_ok=True)
    except Exception:
        pass


def run_desktop_ui(
    agent: LokiAgent | None = None,
    port: int | None = None,
    open_window: bool = True,
    startup_timeout: float = 5.0,
) -> bool:
    """Start local server and launch the LOKI desktop interface.

    Args:
        agent: Optional pre-configured LokiAgent instance.
        port: Preferred port (finds free port if None or if occupied).
        open_window: Whether to launch the desktop browser window.
        startup_timeout: Maximum seconds to wait for server readiness.

    Returns:
        True if server started successfully and ran, False if startup failed.
    """
    # Single-instance detection: check for active server via metadata file or default port
    if port is None:
        active_info = _read_active_server_info()
        active_url = active_info.get("url") if active_info else None

        # Fallback probe to default port if metadata file was absent or not yet created
        if not active_url and _probe_loki_server("http://127.0.0.1:11435"):
            active_url = "http://127.0.0.1:11435"

        if active_url:
            logger.info("Existing LOKI UI server is already active at %s", active_url)
            print(f"\n[INFO] LOKI Desktop UI is already active at {active_url}.", flush=True)
            if open_window:
                launch_desktop_window(active_url)
            return True

    target_port = port if port is not None else find_free_port()
    app = create_ui_app(agent=agent)
    url = f"http://127.0.0.1:{target_port}"

    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=target_port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    # Run Uvicorn in background thread
    server_thread = threading.Thread(
        target=server.run,
        name="LokiUIServerThread",
        daemon=True,
    )
    server_thread.start()

    # Wait for server readiness before opening any browser window
    server_ready = wait_for_server_ready(
        url=url,
        timeout=startup_timeout,
        server_thread=server_thread,
    )

    if not server_ready:
        logger.error("Timed out waiting for LOKI UI server at %s", url)
        print(
            f"\n[ERROR] LOKI Desktop UI failed to start at {url}. "
            f"Server did not become responsive within {startup_timeout}s.",
            file=sys.stderr,
            flush=True,
        )
        server.should_exit = True
        server_thread.join(timeout=2.0)
        return False

    logger.info("LOKI Desktop UI is active at %s", url)
    _write_server_info(url, target_port)

    proc: subprocess.Popen[Any] | None = None
    if open_window:
        proc = launch_desktop_window(url)

    try:
        # Check if browser process exited immediately (Edge / Chrome process delegation)
        if proc:
            time.sleep(1.0)
            if proc.poll() is not None:
                logger.debug(
                    "Browser process PID %s exited with code %s (window delegated to browser manager). Server will remain active.",
                    proc.pid,
                    proc.returncode,
                )
                proc = None

        if proc:
            # Standalone browser process is running; track its lifetime
            while proc.poll() is None:
                time.sleep(0.5)
            logger.info("Browser window closed (PID %s exited with %s).", proc.pid, proc.returncode)
        else:
            # Browser delegated to existing browser instance or system browser opened.
            # Keep server running until user interrupts (Ctrl+C / SIGINT).
            print(f"\n============================================================", flush=True)
            print(f"  LOKI Desktop UI is active: {url}", flush=True)
            print(f"  Press Ctrl+C to shut down.", flush=True)
            print(f"============================================================\n", flush=True)
            while server_thread.is_alive() and not server.should_exit:
                server_thread.join(timeout=0.5)

    except KeyboardInterrupt:
        logger.info("Shutting down LOKI UI on user interrupt...")
    finally:
        _clear_server_info()
        server.should_exit = True
        server_thread.join(timeout=3.0)
        logger.info("LOKI UI server shut down cleanly.")

    return True
