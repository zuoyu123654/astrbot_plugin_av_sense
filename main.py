"""语音视频感知（astrbot_plugin_av_sense）

两个能力收在一个插件里：

1. **语音消息**（Record）→ 下载 → 本机 ASR 转文字 → 注入模型
   （AstrBot 自己已经转过的就不重复转）
2. **视频消息**（Video）→ 只处理**前 3 分钟** → 每 10 秒抽一帧 + 开头一帧 + 最后一帧，
   不足 5 帧就把间隔压短补到 5 帧 → 帧当图片喂给模型看
   → 同时把前 3 分钟的音轨转文字

只依赖：ffmpeg / ffprobe、本机 ASR 服务（D:\\app1\\asr\\asr_server.py，端口 9881）、aiocqhttp。
不调用别的 AstrBot 插件。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

try:  # AstrBot 4.17+
    from astrbot.core.agent.message import TextPart  # type: ignore
except Exception:  # noqa: BLE001
    TextPart = None  # type: ignore[assignment]

PLUGIN_NAME = "astrbot_plugin_av_sense"
PLUGIN_VERSION = "0.3.2"


def _has_whisper(python: str | None) -> bool:
    """那个 python 里有没有 faster_whisper（有就可以不用 funasr 模型）。"""

    if not python:
        return False
    try:
        import subprocess

        r = subprocess.run(
            [python, "-c", "import faster_whisper"],
            capture_output=True,
            timeout=20,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


@register(
    PLUGIN_NAME,
    "paopao",
    "语音视频感知：语音转文字 + 视频抽帧看图（只处理前 3 分钟）",
    PLUGIN_VERSION,
)
class AVSensePlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None):
        super().__init__(context)
        self.context = context
        cfg = config if isinstance(config, dict) else {}
        self.cfg = cfg

        self.enable_voice = bool(cfg.get("enable_voice", True))
        self.enable_video = bool(cfg.get("enable_video", True))
        self.enable_asr = bool(cfg.get("enable_asr", True))
        self.asr_url = str(
            cfg.get("asr_url", "http://127.0.0.1:9881/v1/audio/transcriptions")
        )
        # 视频
        self.max_video_seconds = max(10, int(cfg.get("max_video_seconds", 180) or 180))
        self.frame_step = max(1.0, float(cfg.get("frame_step_seconds", 10) or 10))
        self.min_frames = max(1, int(cfg.get("min_frames", 5) or 5))
        self.max_frames = max(self.min_frames, int(cfg.get("max_frames", 24) or 24))
        # 场景切换补帧 + 相似帧去重
        self.scene_detect = bool(cfg.get("scene_detect", True))
        try:
            self.scene_threshold = float(cfg.get("scene_threshold", 0.3))
        except (TypeError, ValueError):
            self.scene_threshold = 0.3
        try:
            self.dedupe_diff = float(cfg.get("dedupe_diff", 3.0))
        except (TypeError, ValueError):
            self.dedupe_diff = 3.0
        self.frame_side = max(320, int(cfg.get("frame_side", 768) or 768))
        self.max_video_mb = max(1, int(cfg.get("max_video_mb", 200) or 200))
        self.timeout = max(10, int(cfg.get("timeout", 120) or 120))
        # 存下来的图/音频保留几天（默认 7），到期自动删
        try:
            self.retain_days = max(0.0, float(cfg.get("retain_days", 7) or 0))
        except (TypeError, ValueError):
            self.retain_days = 7.0
        self.cleanup_minutes = max(10, int(cfg.get("cleanup_interval_minutes", 60) or 60))
        # 本机 ASR 自动探测（有 funasr 模型 + 带 funasr 的 python 就直接用）
        self.auto_local_asr = bool(cfg.get("auto_local_asr", True))
        self.local_asr_python = str(cfg.get("local_asr_python", "") or "").strip()
        self.local_asr_port = max(1024, int(cfg.get("local_asr_port", 9881) or 9881))
        self.local_asr_engine = str(cfg.get("local_asr_engine", "faster-whisper") or "faster-whisper")
        self.whisper_model = str(cfg.get("whisper_model", "small") or "small")
        self._asr_proc = None
        self._asr_ready = False
        self._asr_checked = False

        self.ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe = shutil.which("ffprobe") or "ffprobe"
        # 临时媒体放插件自己的数据目录（符合数据持久化审计），不再丢系统 temp
        try:
            self.work_dir = Path(StarTools.get_data_dir(PLUGIN_NAME)) / "media"
        except Exception:  # noqa: BLE001
            self.work_dir = Path(__file__).resolve().parent / "data" / "media"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # 把旧版放在系统 temp 里的东西搬过来（7 天清理照旧）
        try:
            legacy = Path(tempfile.gettempdir()) / "av_sense"
            if legacy.is_dir():
                for item in list(legacy.iterdir()):
                    target = self.work_dir / item.name
                    if not target.exists():
                        shutil.move(str(item), str(target))
                logger.info("[AVSense] 已把旧临时目录内容搬到 %s", self.work_dir)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[AVSense] 旧目录搬迁跳过：%s", exc)

    # ── 生命周期：定时清理旧文件 ──────────────────────────────────────
    async def initialize(self) -> None:
        self._cleaner = asyncio.create_task(self._cleanup_loop())
        kept = await self._cleanup_once(verbose=True)
        logger.info(
            "[AVSense] 已启动：保留 %s 天，后台每 %s 分钟清理一次（本次删了 %s 个文件）",
            self.retain_days,
            self.cleanup_minutes,
            kept,
        )

    async def terminate(self) -> None:
        task = getattr(self, "_cleaner", None)
        if task and not task.done():
            task.cancel()
        proc = getattr(self, "_asr_proc", None)
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass

    async def _cleanup_once(self, verbose: bool = False) -> int:
        """删掉超过 retain_days 的文件；返回删除数量。"""

        if self.retain_days <= 0:
            return 0
        cutoff = time.time() - self.retain_days * 86400
        removed = 0

        def _sweep() -> int:
            n = 0
            for p in self.work_dir.rglob("*"):
                try:
                    if p.is_file() and p.stat().st_mtime < cutoff:
                        p.unlink()
                        n += 1
                except OSError:
                    continue
            # 清掉空目录
            for d in sorted(self.work_dir.rglob("*"), reverse=True):
                try:
                    if d.is_dir() and not any(d.iterdir()):
                        d.rmdir()
                except OSError:
                    continue
            return n

        try:
            removed = await asyncio.to_thread(_sweep)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[AVSense] 清理失败：%s", exc)
        if removed and verbose:
            logger.info("[AVSense] 清理了 %s 个超过 %s 天的临时文件", removed, self.retain_days)
        return removed

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cleanup_minutes * 60)
            try:
                await self._cleanup_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[AVSense] 清理循环出错：%s", exc)

    # ── 通用 ──────────────────────────────────────────────────────────
    @staticmethod
    def _find_media(event: AstrMessageEvent, type_name: str) -> Any | None:
        try:
            for comp in event.message_obj.message:
                if type(comp).__name__ == type_name:
                    return comp
        except Exception:  # noqa: BLE001
            return None
        return None

    async def _run(self, *args: str) -> tuple[int, bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        return proc.returncode or 0, out, err

    async def _resolve_file(self, comp: Any, event: AstrMessageEvent, suffix: str) -> str | None:
        """把组件变成本地文件；返回 (路径)。远端 URL 直接下载。"""

        for attr in ("path", "file"):
            val = str(getattr(comp, attr, "") or "")
            if val and Path(val).is_file():
                return val

        url = str(getattr(comp, "url", "") or "")
        if url.startswith("file://"):
            p = Path(url[7:].lstrip("/"))
            if p.is_file():
                return str(p)
        if url.startswith("http"):
            return url  # 交给调用方决定（视频可直接把 url 给 ffmpeg）

        file_id = str(getattr(comp, "file", "") or "")
        if file_id:
            try:
                data = await event.bot.call_action("get_file", file_id=file_id)
                path = (data or {}).get("file") or (data or {}).get("path")
                if path and Path(str(path)).is_file():
                    return str(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[AVSense] get_file 失败：%s", exc)
        return None

    async def _download(self, url: str) -> str | None:
        try:
            dst = self.work_dir / f"dl_{uuid.uuid4().hex}.bin"
            def _get() -> None:
                with urllib.request.urlopen(url, timeout=self.timeout) as r:
                    dst.write_bytes(r.read())
            await asyncio.to_thread(_get)
            if dst.stat().st_size > self.max_video_mb * 1024 * 1024:
                logger.warning("[AVSense] 文件超过 %sMB，跳过", self.max_video_mb)
                dst.unlink(missing_ok=True)
                return None
            return str(dst)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[AVSense] 下载失败：%s", exc)
            return None

    async def _probe(self, path: str) -> dict[str, Any]:
        code, out, _ = await self._run(
            self.ffprobe, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", path,
        )
        if code != 0 or not out:
            return {}
        try:
            return json.loads(out.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return {}

    # ── 本机 ASR：自动探测 + 自动拉起 ────────────────────────────────
    def _find_asr_models(self) -> str | None:
        """找 funasr 风格的模型目录（speech_paraformer-large*）。找不到返回 None。

        注意：Windows 上不能用 Path(".").glob("D:/...")，那不是绝对路径匹配。
        """

        roots: list[Path] = [Path("D:/app1"), Path("C:/app1"), Path("D:/"), Path("C:/")]
        roots += [Path.home()]
        seen: set[str] = set()
        for root in roots:
            if not root.is_dir():
                continue
            try:
                for gsv in list(root.glob("GPT-SoVITS*")) + list(root.glob("*/GPT-SoVITS*")):
                    for sub in (
                        gsv / "tools" / "asr" / "models",
                        gsv / "GPT-SoVITS" / "tools" / "asr" / "models",
                    ):
                        key = str(sub).lower()
                        if key in seen or not sub.is_dir():
                            continue
                        seen.add(key)
                        if any(sub.glob("speech_paraformer-large*")):
                            return str(sub)
            except Exception:  # noqa: BLE001
                continue
        return None

    def _find_asr_python(self) -> str | None:
        """找一个 import 得动 funasr 的 python。"""

        if self.local_asr_python and Path(self.local_asr_python).is_file():
            return self.local_asr_python
        import sys as _sys

        cands: list[str] = []
        for root in (Path("D:/app1"), Path("C:/app1"), Path("D:/"), Path.home()):
            if not root.is_dir():
                continue
            try:
                for gsv in list(root.glob("GPT-SoVITS*")) + list(root.glob("*/GPT-SoVITS*")):
                    for py in (
                        gsv / "runtime" / "python.exe",
                        gsv / "GPT-SoVITS" / "runtime" / "python.exe",
                    ):
                        if py.is_file():
                            cands.append(str(py))
            except Exception:  # noqa: BLE001
                continue
        if cands:
            return cands[0]
        # 实在没有就用 AstrBot 自己的解释器碰碰运气（有些人是全局装的）
        return _sys.executable

    async def _port_open(self, port: int) -> bool:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", port), timeout=1.5
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            return True
        except Exception:  # noqa: BLE001
            return False

    async def _ensure_local_asr(self) -> bool:
        """确保有可用的本机 ASR 服务；返回是否可用（结果缓存，只探测一次）。"""

        if self._asr_ready:
            return True
        if self._asr_checked and not self._asr_ready:
            return False
        self._asr_checked = True

        # 已经在跑（不管是谁起的）
        if await self._port_open(self.local_asr_port):
            self.asr_url = f"http://127.0.0.1:{self.local_asr_port}/v1/audio/transcriptions"
            self._asr_ready = True
            logger.info("[AVSense] 本机 ASR 已在运行（端口 %s）", self.local_asr_port)
            return True

        if not self.auto_local_asr:
            return False

        engine = self.local_asr_engine or "faster-whisper"
        models = self._find_asr_models()
        python = self._find_asr_python()
        script = Path(__file__).resolve().parent / "optional_asr" / "asr_server.py"
        has_whisper = await asyncio.to_thread(_has_whisper, python)
        need_models = engine == "funasr" or (engine == "auto" and not has_whisper)
        if (need_models and not models) or not python or not script.is_file():
            logger.info(
                "[AVSense] 没有可用的本机 ASR（引擎=%s｜模型=%s｜python=%s｜脚本=%s），跳过转写",
                engine, bool(models), python or "无", script.is_file(),
            )
            return False

        try:
            env = dict(**__import__("os").environ)
            env["AVSENSE_ASR_MODELS"] = models
            args = [python, str(script), "--port", str(self.local_asr_port), "--engine", engine]
            if models:
                args += ["--models", models]
            if engine != "funasr":
                args += ["--whisper-model", self.whisper_model]
            self._asr_proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=env,
            )
            logger.info(
                "[AVSense] 正在拉起本机 ASR（引擎 %s，python %s）…", engine, python
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[AVSense] 拉起本机 ASR 失败：%s", exc)
            return False

        for _ in range(60):
            if await self._port_open(self.local_asr_port):
                self.asr_url = f"http://127.0.0.1:{self.local_asr_port}/v1/audio/transcriptions"
                self._asr_ready = True
                logger.info("[AVSense] 本机 ASR 就绪（端口 %s，引擎 %s）", self.local_asr_port, engine)
                return True
            await asyncio.sleep(2)
        logger.warning("[AVSense] 本机 ASR 两分钟内没起来，先不用转写")
        return False

    async def _transcribe(self, src: str, limit: float | None = None) -> str:
        """抽音轨（可选截断前 limit 秒）→ 本机 ASR。"""

        if not self.enable_asr:
            return ""
        wav = self.work_dir / f"a_{uuid.uuid4().hex}.wav"
        args = [self.ffmpeg, "-y", "-loglevel", "error"]
        if limit:
            args += ["-t", f"{limit:.1f}"]
        args += ["-i", src, "-vn", "-ac", "1", "-ar", "16000", str(wav)]
        code, _, err = await self._run(*args)
        if code != 0 or not wav.is_file():
            logger.warning("[AVSense] 抽音轨失败：%s", err[:150])
            return ""
        if wav.stat().st_size < 2048:
            return ""

        boundary = "----av" + uuid.uuid4().hex
        body = b"".join(
            [
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n".encode(),
                (
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                    f"filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
                ).encode(),
                wav.read_bytes(),
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        req = urllib.request.Request(
            self.asr_url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        def _post() -> str:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return str(json.loads(r.read().decode()).get("text") or "")

        # ① 本机识别优先（不花钱；没有可用的本机引擎就跳过这一步）
        if await self._ensure_local_asr() and self.asr_url:
            try:
                text = (await asyncio.to_thread(_post)).strip()
                if text:
                    logger.info("[AVSense] 本机转写成功（%s 字）", len(text))
                    return text
            except Exception as exc:  # noqa: BLE001
                logger.info("[AVSense] 本机识别失败（%s），看框架 STT", exc)

        # ② 兜底：AstrBot 自带的语音转文字 provider（一般没人配，所以放后面）
        try:
            pm = getattr(self.context, "provider_manager", None)
            stt = getattr(pm, "curr_stt_provider_inst", None) if pm else None
            if stt is not None:
                text = await stt.get_text(str(wav))
                if text:
                    logger.info("[AVSense] 用框架 STT 转写成功（%s 字）", len(text))
                    return str(text).strip()
        except Exception as exc:  # noqa: BLE001
            logger.info("[AVSense] 框架 STT 也不可用（%s）", exc)
        return ""

    # ── 抽帧计划 ──────────────────────────────────────────────────────
    def plan_timestamps(self, duration: float) -> list[float]:
        """开头一帧 + 每 10 秒（到 3 分钟）+ 最后一帧；不足 5 帧就压短间隔补到 5 帧。"""

        limit = max(0.5, min(duration, float(self.max_video_seconds)))
        last = max(0.0, limit - 0.4)

        ts: list[float] = [0.0]
        t = self.frame_step
        while t < last - 0.5:
            ts.append(round(t, 2))
            t += self.frame_step
        ts.append(round(last, 2))

        if len(ts) < self.min_frames:
            # 视频太短：按需要压缩间隔，保证 min_frames 帧
            span = last if last > 0 else 0.5
            step = span / max(1, self.min_frames - 1)
            ts = [round(i * step, 2) for i in range(self.min_frames)]
            ts[-1] = round(last, 2)

        out: list[float] = []
        for v in ts:
            if not out or abs(v - out[-1]) > 0.05:
                out.append(v)
        return out[: self.max_frames]

    async def _scene_times(self, src: str, limit: float) -> list[float]:
        """用 ffmpeg 的 scene 检测找出「换画面」的时间点（只扫前 limit 秒）。"""

        if not self.scene_detect:
            return []
        code, _, err = await self._run(
            self.ffmpeg, "-v", "info", "-t", f"{limit:.1f}", "-i", src,
            "-vf", f"select='gt(scene,{self.scene_threshold})',showinfo",
            "-an", "-f", "null", "-",
        )
        if code != 0 or not err:
            return []
        times: list[float] = []
        for line in err.decode("utf-8", "replace").splitlines():
            if "pts_time:" not in line:
                continue
            try:
                v = float(line.split("pts_time:")[1].split()[0])
            except (IndexError, ValueError):
                continue
            if 0.3 < v < limit - 0.3:
                times.append(round(v, 2))
        return times

    @staticmethod
    def _diff(prev: str, cur: str) -> float:
        """两张图有多不一样（0~255 的灰度均差）。没有 PIL 就返回大数（= 当成不一样）。"""

        try:
            from PIL import Image, ImageChops, ImageStat

            with Image.open(prev) as a, Image.open(cur) as b:
                ga = a.convert("L").resize((32, 32))
                gb = b.convert("L").resize((32, 32))
                return float(ImageStat.Stat(ImageChops.difference(ga, gb)).mean[0])
        except Exception:  # noqa: BLE001
            return 999.0

    async def _frames(self, src: str, duration: float, limit: float | None = None) -> list[str]:
        ts = self.plan_timestamps(duration)
        limit_v = float(limit or self.max_video_seconds)
        # 场景切换帧补进来（和基准帧合并，间隔太近的丢掉）
        if self.scene_detect:
            try:
                extra = await self._scene_times(src, limit_v)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[AVSense] 场景检测失败，只用基准帧：%s", exc)
                extra = []
            if extra:
                merged = sorted(set(round(x, 2) for x in ts) | set(extra))
                picked: list[float] = []
                for v in merged:
                    if not picked or v - picked[-1] >= 1.0:
                        picked.append(v)
                # 太多就均匀抽稀（保留开头和结尾）
                if len(picked) > self.max_frames:
                    keep = [picked[0]]
                    step = (len(picked) - 2) / max(1, self.max_frames - 2)
                    i = 1.0
                    while len(keep) < self.max_frames - 1:
                        keep.append(picked[int(round(i))])
                        i += step
                    keep.append(picked[-1])
                    picked = sorted(set(keep))
                ts = picked
                logger.info("[AVSense] 基准 + 场景切换共 %s 个时间点", len(ts))

        dst_dir = self.work_dir / f"f_{uuid.uuid4().hex}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        for i, sec in enumerate(ts):
            dst = dst_dir / f"frame_{i:02d}_{sec:.1f}s.jpg"
            args = [self.ffmpeg, "-y", "-loglevel", "error", "-ss", f"{sec:.2f}"]
            if limit:
                args += ["-t", f"{limit:.1f}"]
            args += [
                "-i", src, "-frames:v", "1",
                "-vf", f"scale='min({self.frame_side},iw)':-2", str(dst),
            ]
            code, _, err = await self._run(*args)
            if code != 0 or not dst.is_file() or dst.stat().st_size == 0:
                logger.debug("[AVSense] 抽帧失败 @%.2fs: %s", sec, err[:120])
                continue
            # 和上一张几乎一样就丢掉（画面没变）
            if files and self._diff(files[-1], str(dst)) < self.dedupe_diff:
                dst.unlink(missing_ok=True)
                continue
            files.append(str(dst))
        # 去重后如果不足 min_frames，从基准帧里补齐（避免只剩两张图）
        if len(files) < self.min_frames:
            for sec in self.plan_timestamps(duration):
                if len(files) >= self.min_frames:
                    break
                dst = dst_dir / f"pad_{sec:.1f}s.jpg"
                args = [self.ffmpeg, "-y", "-loglevel", "error", "-ss", f"{sec:.2f}"]
                if limit:
                    args += ["-t", f"{limit:.1f}"]
                args += [
                    "-i", src, "-frames:v", "1",
                    "-vf", f"scale='min({self.frame_side},iw)':-2", str(dst),
                ]
                code, _, _ = await self._run(*args)
                if code == 0 and dst.is_file() and dst.stat().st_size > 0:
                    if str(dst) not in files and all(
                        self._diff(f, str(dst)) >= self.dedupe_diff for f in files
                    ):
                        files.append(str(dst))
        # 去重太狠（画面基本不动）⇒ 硬补到 min_frames，不再做相似判断
        if len(files) < self.min_frames:
            for sec in self.plan_timestamps(duration):
                if len(files) >= self.min_frames:
                    break
                dst = dst_dir / f"force_{sec:.1f}s.jpg"
                args = [self.ffmpeg, "-y", "-loglevel", "error", "-ss", f"{sec:.2f}"]
                if limit:
                    args += ["-t", f"{limit:.1f}"]
                args += [
                    "-i", src, "-frames:v", "1",
                    "-vf", f"scale='min({self.frame_side},iw)':-2", str(dst),
                ]
                code, _, _ = await self._run(*args)
                if code == 0 and dst.is_file() and dst.stat().st_size > 0:
                    files.append(str(dst))
        return files

    def _inject_text(self, req: Any, note: str) -> None:
        if TextPart is None:
            return
        try:
            part = TextPart(text=note)
            for attr in ("make_as_temp", "mark_as_temp"):
                marker = getattr(part, attr, None)
                if callable(marker):
                    part = marker()
                    break
            req.extra_user_content_parts.append(part)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[AVSense] 注入失败：%s", exc)

    # ── 主钩子 ────────────────────────────────────────────────────────
    @filter.on_llm_request()
    async def inject_media(self, event: AstrMessageEvent, req: Any):
        if getattr(event, "_av_sense_done", False):
            return
        try:
            setattr(event, "_av_sense_done", True)
        except Exception:  # noqa: BLE001
            pass

        # ① 视频
        if self.enable_video:
            video = self._find_media(event, "Video")
            if video is not None:
                await self._handle_video(video, event, req)
                return

        # ② 语音（AstrBot 自己已经转出文字的话就不再重复）
        if self.enable_voice:
            record = self._find_media(event, "Record")
            if record is not None:
                await self._handle_voice(record, event, req)

    async def _handle_voice(self, comp: Any, event: AstrMessageEvent, req: Any) -> None:
        existing = str(getattr(event, "message_str", "") or "").strip()
        if existing:  # 已经有文字（框架的 STT 转过），不重复
            return
        started = time.time()
        src = await self._resolve_file(comp, event, ".silk")
        if not src:
            logger.warning("[AVSense] 语音：取不到文件")
            return
        if src.startswith("http"):
            local = await self._download(src)
            if not local:
                return
            src = local
        text = await self._transcribe(src)
        logger.info("[AVSense] 语音 %.1fKB → %s 字｜耗时 %.1fs", Path(src).stat().st_size / 1024, len(text), time.time() - started)
        if text:
            try:
                event.message_str = text
            except Exception:  # noqa: BLE001
                pass
            self._inject_text(
                req,
                f"<voice_note>老大刚才发的是语音，转成文字是：「{text}」。"
                "按这句话回答他。</voice_note>",
            )
        else:
            self._inject_text(
                req,
                "<voice_note>老大刚才发了一条语音，但这台机器的识别服务没听清（可能是噪音、方言或服务没开）。"
                "用你自己的话让他知道你没听清、请他再说一遍或者打字，别假装听到了。</voice_note>",
            )

    async def _handle_video(self, comp: Any, event: AstrMessageEvent, req: Any) -> None:
        started = time.time()
        src = await self._resolve_file(comp, event, ".mp4")
        if not src:
            logger.warning("[AVSense] 视频：取不到文件")
            return

        info = await self._probe(src)
        duration = 0.0
        try:
            duration = float(info.get("format", {}).get("duration") or 0)
        except (TypeError, ValueError):
            duration = 0.0
        has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))

        work_src = src
        if src.startswith("http"):
            # 远端：本地能直接抽就直接抽（还省一次整包下载）
            local = await self._download(src)
            work_src = local or src

        limit = float(min(duration or self.max_video_seconds, self.max_video_seconds))
        frames = await self._frames(work_src, duration or limit, limit)
        transcript = await self._transcribe(work_src, limit) if has_audio else ""
        size_mb = 0.0
        try:
            if not work_src.startswith("http"):
                size_mb = Path(work_src).stat().st_size / 1048576
        except OSError:
            size_mb = 0.0

        logger.info(
            "[AVSense] 视频 时长%.1fs/处理%.0fs/%s音轨｜%.1fMB｜抽到 %s 帧｜转写 %s 字｜耗时 %.1fs",
            duration, limit, "有" if has_audio else "无",
            size_mb, len(frames), len(transcript), time.time() - started,
        )

        if frames:
            req.image_urls = list(getattr(req, "image_urls", None) or []) + frames

        time_desc = (
            f"整条约 {duration:.0f} 秒，你只看/听了**前 {limit:.0f} 秒**"
            if duration > limit
            else f"约 {duration:.0f} 秒"
        )
        note = (
            f"<video_note>老大发来一条视频（{time_desc}），"
            f"已经抽出 {len(frames)} 张关键帧给你看（按时间先后）。"
        )
        if transcript:
            note += f"前 {limit:.0f} 秒里的声音转成文字是：「{transcript}」"
        elif has_audio:
            note += "这段有声音，但没识别到人声（可能是纯音乐/环境音）。"
        else:
            note += "这条视频没有音轨。"
        note += "就用这些来回答他，别假装没看见视频。</video_note>"
        self._inject_text(req, note)
