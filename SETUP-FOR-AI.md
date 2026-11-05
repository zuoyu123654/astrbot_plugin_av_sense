# 给 AI 的施工文档：把「语音视频感知」配到能用

> 读者是**接手配置的 AI**。使用者会把这份文档 + `README.md` 给你，并说：
> 「请帮我配置好 AstrBot 里的【语音视频感知】插件，并检测功能是否可用。」
>
> 目标：**发语音能听懂、发视频能看懂**（画面 + 声音），并且 7 天后自动清掉临时文件。
> 本文所有步骤都是真机验证过的，按顺序做。

---

## 0. 先问使用者两件事

1. 主模型**能不能看图**（多模态）？——视频要靠它。不能的话，先把这一步告诉他：
   视频会退化成「只听得见声音」，得换一个支持图片输入的模型。
2. 语音识别走哪条：**AstrBot 自带**（推荐，配好就能用）还是**自建 ASR 服务**（离线、不花钱）。

---

## 1. 装插件

```
<AstrBot>/data/plugins/astrbot_plugin_av_sense/
├── main.py
├── metadata.yaml
├── _conf_schema.json
├── README.md
└── SETUP-FOR-AI.md
```

放好后**完整重启 AstrBot**（面板「重载插件」不重读配置）。启动日志应当有：

```
Loading plugin astrbot_plugin_av_sense ...
[AVSense] 已启动：保留 7.0 天，后台每 60 分钟清理一次（本次删了 0 个文件）
```

报 `ModuleNotFoundError` / `SyntaxError` ⇒ 文件没放全或版本不对；报 `is disabled` ⇒ 插件在停用名单
（`data/data_v4.db` 的 `preferences` → `inactivated_plugins`），启用后再重启。

---

## 2. 装 ffmpeg（必需）

抽帧/抽音轨全靠它。验证：

```powershell
ffmpeg -version
ffprobe -version
```

**两个都要有**（只有 ffmpeg 是不够的，`ffprobe` 用来读时长和音轨信息）。
没有就把 ffmpeg full build 的 bin 目录加进系统 PATH，然后**重启 AstrBot**（新进程才继承新 PATH）。

---

## 3. 配语音识别（二选一）

### 3.1 用 AstrBot 自带的（推荐）

面板 → 配置 → **语音转文字**：选一个 provider（Whisper API / MiMo STT / 自建 whisper 服务等），填 key 和地址，
**把它设成默认 STT**。配置层面对应：

```json
"provider_stt_settings": { "enable": true, "provider_id": "<你的 STT provider id>" }
```

插件会自动用它转写**语音消息**和**视频音轨**（日志：`[AVSense] 用框架 STT 转写成功（N 字）`）。

> ⚠️ 用 `openai_whisper_api` 这类适配器时，注意它在 provider 条目里读的是 **`api_key`** 字段
> （不是 `key` 数组），否则会报 `Missing credentials`。

### 3.2 自建 ASR（离线、不花钱）

本机用过一套 GPT-SoVITS 自带 funasr（中文 paraformer）的 OpenAI 兼容服务：

- 需要：`funasr` + 三个模型目录（`speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`、
  `speech_fsmn_vad_zh-cn-16k-common-pytorch`、`punc_ct-transformer_zh-cn-common-vocab272727-pytorch`）
- 起服务（示例）：`python asr_server.py --port 9881`，暴露 `POST /v1/audio/transcriptions`，返回 `{"text": "..."}`
- 插件侧只要把 `asr_url` 指过去（默认就是 `http://127.0.0.1:9881/v1/audio/transcriptions`）

没装成也没关系：**视频的画面部分照样能用**，只是听不见声音。

---

## 4. 让主模型能看图（视频必需）

抽出来的帧是当图片塞进请求的（`req.image_urls`）。确认主模型支持图片输入，例如：

```json
"provider": [{ "id": "...", "model": "<多模态模型>", "modalities": [], ... }]
```

> 小坑：`modalities: []` 在 AstrBot 里表示「不限制模态」，不等于「不支持图片」。
> 真不能看图的模型，会在收到图片时报错或直接忽略——那就必须换模型。

自检：把任意一张图片发给机器人，她能说出画面内容 ⇒ 多模态通了。

---

## 5. 验收（照顺序做）

### 5.1 语音

给他发一条 3~5 秒的语音。期望日志：

```
[AVSense] 语音 12.3KB → 5 字｜耗时 1.2s
```

并且她按语音内容回答。若回「没听清」⇒ 回第 3 节。

### 5.2 视频

发一条**短一点、有人说话**的视频（比如 20 秒）。期望日志：

```
[AVSense] 基准 + 场景切换共 N 个时间点        （开了场景检测才有）
[AVSense] 视频 时长20.0s/处理20s/有音轨｜2.1MB｜抽到 5 帧｜转写 23 字｜耗时 5.8s
```

判定：**帧数 ≥ 5**、**转写字数 > 0**、她说的内容能对上视频。

### 5.3 长视频

发一条超过 3 分钟的视频，确认日志里 `处理180s`，并且不会卡住（帧数封顶 24）。

### 5.4 清理

盯一眼临时目录（Windows：`%TEMP%\av_sense`）：里面的帧/音频文件超过 `retain_days` 天会被删掉；
启动日志里那句「保留 7.0 天，后台每 60 分钟清理一次」就是它。

---

## 6. 排查表

| 症状 | 原因 | 修法 |
| --- | --- | --- |
| 日志 `视频：取不到文件` | 协议端不给文件 | 确认 aiocqhttp 支持 `get_file`，或在协议端打开「下载媒体文件」 |
| `抽到 0 帧` | ffmpeg 不在 PATH / 视频编码不支持 | `ffmpeg -version` 测；把 bin 加进 PATH 后重启 |
| 帧数只有 1~2 张 | 画面几乎不动 | 正常（去重了）；想强制 5 帧就把 `dedupe_diff` 调小到 0 |
| `转写 0 字` | STT 没配 / 服务没起 | 回第 3 节；看日志是 `框架 STT 不可用` 还是 `语音识别失败` |
| 她说「没听清」 | STT 全挂 | 同上 |
| 她只描述画面、不提内容 | 音轨没转写 | 同上 |
| 很慢 | 帧多/视频长 | `max_frames` 调小、`frame_side` 调小、`frame_step_seconds` 调大 |
| 磁盘涨 | 清理被关了 | 看 `retain_days` 是不是 0 |

---

## 7. 配完要告诉使用者什么（照抄改数字）

> 配好了，现在：
> - 语音：**能听懂**（走 <框架 STT / 本机 ASR>），听不清会直接说没听清
> - 视频：**只看前 3 分钟**，开头 + 每 10 秒 + 结尾各抽一帧，换镜头会补帧，不足 5 帧自动补到 5 帧；
>   画面交给模型看，声音转成文字
> - 清理：临时文件保留 **7 天**，到期自动删（`retain_days` 可改）
> - 依赖：只需要 ffmpeg/ffprobe；不调用其它插件
