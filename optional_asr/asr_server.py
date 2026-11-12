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

# 模型目录来源：--models > AVSENSE_ASR_MODELS > AVSENSE_GSV_HOME/tools/asr/models
# 这里不写死任何本机路径。
DEFAULT_GSV = None
ENGINE = "auto"
WHISPER_MODEL = "small"
MODELS: Path | None = None
ASR: Path | None = None
VAD: Path | None = None
PUNC: Path | None = None

_model = None
_engine = "未加载"  # "faster-whisper" / "funasr"
_lock = threading.Lock()


def resolve_models(path: str | None = None) -> Path:
    """确定模型根目录，并找出三个子模型。"""

    global MODELS, ASR, VAD, PUNC
    import os

    cand = path or os.environ.get("AVSENSE_ASR_MODELS") or ""
    if cand:
        root = Path(cand)
    else:
        gsv_home = os.environ.get("AVSENSE_GSV_HOME", "").strip()
        if not gsv_home:
            raise SystemExit(
                "[asr] 没指定模型目录：用 --models 指定 models 目录，或设环境变量 "
                "AVSENSE_ASR_MODELS（models 目录）/"
                "AVSENSE_GSV_HOME（GPT-SoVITS 根目录）。"
            )
        root = Path(gsv_home) / "tools" / "asr" / "models"
    if not root.is_dir():
        raise SystemExit(f"[asr] 模型目录不存在：{root}")
    asr = next(iter(root.glob("speech_paraformer-large*")), None)
    vad = next(iter(root.glob("speech_fsmn_vad*")), None)
    punc = next(iter(root.glob("punc_ct-transformer*")), None)
    if not (asr and vad):
        raise SystemExit(f"[asr] 模型目录里缺 paraformer/vad：{root}")
    MODELS, ASR, VAD, PUNC = root, asr, vad, punc
    print(f"[asr] 模型根目录：{root}", flush=True)
    return root


def _load_faster_whisper(model_name: str):
    """默认引擎：faster-whisper（国内走 hf-mirror，不然下不动）。"""

    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from faster_whisper import WhisperModel

    print(f"[asr] 加载 faster-whisper（{model_name}）…", flush=True)
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def _load_funasr():
    from funasr import AutoModel

    print("[asr] 加载 funasr paraformer …", flush=True)
    kwargs = {"model": str(ASR), "vad_model": str(VAD), "disable_update": True}
    if PUNC is not None:
        kwargs["punc_model"] = str(PUNC)
    return AutoModel(**kwargs)


def get_model():
    global _model, _engine
    with _lock:
        if _model is not None:
            return _model
        want = ENGINE
        if want in ("auto", "faster-whisper"):
            try:
                _model = _load_faster_whisper(WHISPER_MODEL)
                _engine = "faster-whisper"
                print(f"[asr] 引擎就绪：faster-whisper / {WHISPER_MODEL}", flush=True)
                return _model
            except Exception as exc:  # noqa: BLE001
                if want == "faster-whisper":
                    raise
                print(f"[asr] faster-whisper 不可用（{exc}），改用 funasr", flush=True)
        _model = _load_funasr()
        _engine = "funasr"
        print("[asr] 引擎就绪：funasr paraformer", flush=True)
        return _model


def transcribe(path: Path) -> str:
    model = get_model()
    if _engine == "faster-whisper":
        segments, _info = model.transcribe(str(path), language="zh", vad_filter=True)
        return "".join(seg.text for seg in segments).strip()
    res = model.generate(input=str(path), batch_size_s=300)
    if not res:
        return ""
    return str(res[0].get("text", "") or "").strip()


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
            self._json(200, {"status": "ok", "engine": _engine})
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
    ap.add_argument("--models", default="", help="funasr 模型根目录（含 paraformer/vad/punc）")
    ap.add_argument("--engine", default="auto", choices=["auto", "faster-whisper", "funasr"])
    ap.add_argument("--whisper-model", default="small", help="faster-whisper 的模型名或本地目录")
    args = ap.parse_args()

    global ENGINE, WHISPER_MODEL
    ENGINE = args.engine
    WHISPER_MODEL = args.whisper_model

    # funasr 模型只在需要时才有要求
    try:
        resolve_models(args.models or None)
    except SystemExit:
        if ENGINE == "funasr":
            raise
        print("[asr] 没找到 funasr 模型，只用 faster-whisper", flush=True)
    if args.preload:
        get_model()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[asr] 监听 http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
