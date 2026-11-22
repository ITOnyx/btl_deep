"""
Chạy ứng dụng tra cứu như một app Windows: cửa sổ riêng (pywebview + WebView2), không cần mở trình duyệt.

- Khởi động server Streamlit ẩn trên một cổng trống của 127.0.0.1 (không mở ra mạng ngoài).
- Hiện màn hình chờ trong lúc server khởi động; trang tự nạp khi server sẵn sàng.
- Đóng cửa sổ -> tắt server. Chỉ cho chạy một phiên cùng lúc.
- Log: %LOCALAPPDATA%/FashionSearch/logs/desktop.log (log server: streamlit.log cùng thư mục).

Chạy: pythonw app/desktop.py   (hoặc dùng shortcut tạo bởi scripts/create_desktop_shortcut.ps1)
"""
from __future__ import annotations

import ctypes
import html
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_TITLE = "Tra cứu sản phẩm thời trang"
APP_ID = "HocSau.FashionSearch.Desktop"
ICON_PATH = ROOT / "app" / "assets" / "app.ico"
SERVER_START_TIMEOUT_S = 120  # chỉ chờ server web; mô hình được nạp sau, khi trang mở lần đầu
LOG_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "FashionSearch" / "logs"

log = logging.getLogger("desktop")


def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_DIR / "desktop.log", level=logging.INFO, encoding="utf-8",
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _message_box(text: str, title: str = APP_TITLE, error: bool = True) -> None:
    """Hộp thoại Windows (dùng được cả khi chạy bằng pythonw, không có console)."""
    try:
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10 if error else 0x40)
    except Exception:
        print(f"{title}: {text}", file=sys.stderr)


def _single_instance_lock():
    """Mutex có tên: phiên thứ hai sẽ thấy ERROR_ALREADY_EXISTS (183) và thoát."""
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, f"Local\\{APP_ID}")
    if ctypes.windll.kernel32.GetLastError() == 183:
        return None
    return handle


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int) -> tuple[subprocess.Popen, Path]:
    server_log = LOG_DIR / "streamlit.log"
    out = open(server_log, "w", encoding="utf-8")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "main.py"),
        "--server.port", str(port), "--server.address", "127.0.0.1",
        "--server.headless", "true", "--browser.gatherUsageStats", "false",
        "--global.developmentMode", "false",
    ]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    log.info("Khởi động server: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT, env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return proc, server_log


def _wait_healthy(proc: subprocess.Popen, port: int, timeout_s: float) -> bool:
    url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False  # server chết sớm (thiếu thư viện, lỗi cú pháp...)
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    log.info("Tắt server (pid %s)", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()


def _tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _set_window_icon(title: str) -> None:
    """WinForms của pywebview lấy icon của pythonw.exe; đặt lại icon cửa sổ/taskbar bằng WM_SETICON."""
    if not ICON_PATH.exists():
        return
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return
    for size, which in ((16, 0), (32, 1)):  # ICON_SMALL=0, ICON_BIG=1
        hicon = user32.LoadImageW(None, str(ICON_PATH), 1, size, size, 0x10)  # IMAGE_ICON, LR_LOADFROMFILE
        if hicon:
            user32.SendMessageW(hwnd, 0x80, which, hicon)  # WM_SETICON


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
background:#0F1117;color:#E6E8EB;font-family:'Segoe UI',sans-serif}}
.box{{max-width:640px;text-align:center;padding:24px}} h1{{font-size:20px;font-weight:600;margin:0 0 12px}}
p{{color:#8B949E;font-size:14px;line-height:1.6}} pre{{text-align:left;background:#181B22;color:#C9D1D9;
padding:12px;border-radius:6px;font-size:12px;white-space:pre-wrap;max-height:240px;overflow:auto}}
.spin{{width:36px;height:36px;border:3px solid #2A2F3A;border-top-color:#4C8BF5;border-radius:50%;
margin:0 auto 20px;animation:s 1s linear infinite}} @keyframes s{{to{{transform:rotate(360deg)}}}}
</style></head><body><div class="box">{body}</div></body></html>"""

SPLASH_HTML = _PAGE.format(body=(
    '<div class="spin"></div><h1>Đang khởi động…</h1>'
    "<p>Lần đầu mở có thể mất khoảng một phút để nạp mô hình SigLIP và index.</p>"
))


def _error_html(msg: str, detail: str) -> str:
    return _PAGE.format(body=(
        f"<h1>Không khởi động được ứng dụng</h1><p>{html.escape(msg)}</p>"
        + (f"<pre>{html.escape(detail)}</pre>" if detail else "")
        + f"<p>Log: {html.escape(str(LOG_DIR))}</p>"
    ))


def main() -> int:
    _setup_logging()
    try:
        import webview
    except ImportError:
        _message_box("Thiếu thư viện pywebview.\nCài bằng lệnh:  pip install pywebview")
        return 1

    lock = _single_instance_lock()
    if lock is None:
        _message_box("Ứng dụng đang chạy rồi.", error=False)
        return 0

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)  # icon riêng trên taskbar
    except Exception:
        pass

    port = _free_port()
    proc, server_log = _start_server(port)
    window = webview.create_window(APP_TITLE, html=SPLASH_HTML, width=1360, height=880, min_size=(900, 600),
                                   background_color="#0F1117")

    def on_start() -> None:
        _set_window_icon(APP_TITLE)
        if _wait_healthy(proc, port, SERVER_START_TIMEOUT_S):
            log.info("Server sẵn sàng ở cổng %s", port)
            window.load_url(f"http://127.0.0.1:{port}/")
        else:
            code = proc.poll()
            msg = (f"Server dừng đột ngột (mã thoát {code})." if code is not None
                   else f"Server không phản hồi sau {SERVER_START_TIMEOUT_S} giây.")
            log.error("%s", msg)
            window.load_html(_error_html(msg, _tail(server_log)))

    window.events.closed += lambda: _stop_server(proc)
    try:
        webview.start(on_start, gui="edgechromium", private_mode=False,
                      storage_path=str(LOG_DIR.parent / "webview"))
    except Exception as e:
        log.exception("Lỗi cửa sổ ứng dụng")
        _message_box(f"Không mở được cửa sổ ứng dụng (cần Microsoft Edge WebView2 Runtime).\n\n{e}")
        return 1
    finally:
        _stop_server(proc)
        if lock:
            ctypes.windll.kernel32.CloseHandle(lock)
    return 0


if __name__ == "__main__":
    sys.exit(main())
