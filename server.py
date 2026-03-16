#!/usr/bin/env python3
"""
Agent Roadmap 本地笔记服务
用法：python server.py
然后用 Chrome/Edge 打开 http://localhost:8765
"""

import json
import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# ── 配置 ──────────────────────────────────────────
PORT      = 8765
DATA_FILE = "notes.json"   # 与 server.py / html 同级目录
HTML_FILE = "agent-engineer-roadmap.html"
# ──────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.resolve()
DATA_PATH = BASE_DIR / DATA_FILE


def load_notes() -> dict:
    if DATA_PATH.exists():
        try:
            return json.loads(DATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_notes(data: dict):
    DATA_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


class Handler(SimpleHTTPRequestHandler):
    # 静态文件从 BASE_DIR 提供
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def log_message(self, fmt, *args):
        # args[0] 可能是 HTTPStatus 对象，转成字符串再判断
        first = str(args[0]) if args else ""
        if "/api/" in first:
            print(f"  {first}  →  {str(args[1]) if len(args) > 1 else ''}")

    def do_OPTIONS(self):
        self._cors(200)

    def do_GET(self):
        parsed = urlparse(self.path)

        # 忽略 favicon 请求，避免报错
        if parsed.path == '/favicon.ico':
            self._cors(204)
            return

        if parsed.path == "/api/notes":
            data = load_notes()
            self._json(200, data)
        else:
            # 其余交给 SimpleHTTPRequestHandler 提供静态文件
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/notes":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
                save_notes(data)
                self._json(200, {"ok": True})
            except Exception as e:
                self._json(400, {"error": str(e)})
        else:
            self._cors(404)

    # ── helpers ──────────────────────────────────
    def _cors(self, code: int):
        self.send_response(code)
        self._cors_headers()
        self.end_headers()

    def _json(self, code: int, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def main():
    # 检查 html 文件是否存在
    if not (BASE_DIR / HTML_FILE).exists():
        print(f"⚠️  未找到 {HTML_FILE}，请确保与 server.py 放在同一目录")

    # 如果 notes.json 不存在，创建空文件
    if not DATA_PATH.exists():
        save_notes({})
        print(f"✅ 已创建 {DATA_FILE}")

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}/{HTML_FILE}"

    print()
    print("=" * 50)
    print("  Agent Roadmap 本地服务已启动")
    print(f"  请用 Chrome/Edge 打开：")
    print(f"  👉  {url}")
    print()
    print(f"  笔记保存位置：{DATA_PATH}")
    print("  按 Ctrl+C 停止服务")
    print("=" * 50)
    print()

    # 尝试自动打开浏览器
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        sys.exit(0)


if __name__ == "__main__":
    main()