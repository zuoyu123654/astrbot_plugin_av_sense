"""本地语音识别服务（OpenAI Whisper 兼容），给 AstrBot 当 STT 用。

- 模型：GPT-SoVITS 自带的 funasr paraformer（中文，含 VAD + 标点），完全离线
- 接口：POST /v1/audio/transcriptions（multipart，字段 file / model），返回 {"text": "..."}
- 另有一个 GET /health 方便探活

启动（用 GPT-SoVITS 的 runtime python，它装了 funasr）：
    <GPT-SoVITS>/runtime/python.exe asr_server.py --port 9881
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GSV = Path(r"D:\app1\GPT-SoVITS-v2pro-20250604-nvidia50\GPT-SoVITS-v2pro-20250604-nvidia50")
MODELS = GSV / "tools" / "asr" / "models"
ASR = MODELS / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
VAD = MODELS / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC = MODELS / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    with _lock:
        if _model is None:
            from funasr import AutoModel

            print("[asr] 加载 funasr paraformer …", flush=True)
            _model = AutoModel(
                model=str(ASR),
                vad_model=str(VAD),
                punc_model=str(PUNC),
                disable_update=True,
            )
            print("[asr] 模型就绪", flush=True)
    return _model


def transcribe(path: Path) -> str:
    model = get_model()
    res = model.generate(input=str(path), batch_size_s=300)
    if not res:
        return ""
    text = str(res[0].get("text", "") or "").strip()
    return text


def parse_multipart(body: bytes, content_type: str) -> dict[str, bytes]:
    """极简 multipart 解析：返回 {字段名: 字节}。"""

    m = re.search(r"boundary=([^;]+)", content_type or "")
    if not m:
        return {}
    boundary = ("--" + m.group(1).strip().strip('"')).encode()
    out: dict[str, bytes] = {}
    for part in body.split(boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        head_text = head.decode("utf-8", "replace")
        name_m = re.search(r'name="([^"]+)"', head_text)
        if not name_m:
            continue
        out[name_m.group(1)] = data.rstrip(b"\r\n")
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003
        print("[asr] " + fmt % args, flush=True)

    def _json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            self._json(200, {"status": "ok", "engine": "funasr-paraformer"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if not self.path.startswith("/v1/audio/transcriptions"):
            self._json(404, {"error": "not found"})
            return

        audio = b""
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" in ctype:
            fields = parse_multipart(body, ctype)
            audio = fields.get("file") or next(
                (v for k, v in fields.items() if k not in ("model", "language", "prompt")), b""
            )
        else:
            audio = body
        if not audio:
            self._json(400, {"error": "no audio"})
            return

        suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio)
            tmp_path = Path(tmp.name)
        try:
            text = transcribe(tmp_path)
            print(f"[asr] {len(audio) // 1024} KB → {text!r}", flush=True)
            self._json(200, {"text": text})
        except Exception as exc:  # noqa: BLE001
            print("[asr] 识别失败:", exc, flush=True)
            self._json(500, {"error": str(exc)})
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9881)
    ap.add_argument("--preload", action="store_true", help="启动时就加载模型")
    args = ap.parse_args()

    if args.preload:
        get_model()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[asr] 监听 http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
