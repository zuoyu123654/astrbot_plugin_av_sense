"""QQ 表情（小黄脸 Face / 商城表情 mface）代码与语义映射表。"""

from __future__ import annotations

# QQ 经典表情 ID 到中文含义的映射表
QQ_FACE_MAP: dict[int, str] = {
    0: "惊讶",
    1: "撇嘴",
    2: "色",
    3: "发呆",
    4: "得意",
    5: "流泪",
    6: "害羞",
    7: "闭嘴",
    8: "睡",
    9: "大哭",
    10: "尴尬",
    11: "发怒",
    12: "调皮",
    13: "呲牙",
    14: "微笑",
    15: "难过",
    16: "酷",
    17: "冷汗",
    18: "抓狂",
    19: "吐",
    20: "偷笑",
    21: "可爱",
    22: "白眼",
    23: "傲慢",
    24: "饥饿",
    25: "困",
    26: "惊恐",
    27: "流汗",
    28: "憨笑",
    29: "悠闲",
    30: "奋斗",
    31: "咒骂",
    32: "疑问",
    33: "嘘",
    34: "晕",
    35: "折磨",
    36: "衰",
    37: "骷髅",
    38: "敲打",
    39: "再见",
    40: "擦汗",
    41: "抠鼻",
    42: "鼓掌",
    43: "糗大了",
    44: "坏笑",
    45: "左哼哼",
    46: "右哼哼",
    47: "哈欠",
    48: "鄙视",
    49: "委屈",
    50: "快哭了",
    51: "阴险",
    52: "亲亲",
    53: "吓",
    54: "可怜",
    55: "菜刀",
    56: "西瓜",
    57: "啤酒",
    58: "篮球",
    59: "乒乓",
    60: "咖啡",
    61: "饭",
    62: "猪头",
    63: "玫瑰",
    64: "凋谢",
    65: "示爱",
    66: "爱心",
    67: "心碎",
    68: "蛋糕",
    69: "闪电",
    70: "炸弹",
    71: "刀",
    72: "足球",
    73: "瓢虫",
    74: "便便",
    75: "月亮",
    76: "太阳",
    77: "礼物",
    78: "拥抱",
    79: "强",
    80: "弱",
    81: "握手",
    82: "胜利",
    83: "抱拳",
    84: "勾引",
    85: "拳头",
    86: "差劲",
    87: "爱你",
    88: "NO",
    89: "OK",
    90: "爱情",
    91: "飞吻",
    92: "跳跳",
    93: "发抖",
    94: "怄火",
    95: "转圈",
    96: "磕头",
    97: "回头",
    98: "跳绳",
    99: "挥手",
    100: "激动",
    101: "街舞",
    102: "献吻",
    103: "左太极",
    104: "双喜",
    105: "鞭炮",
    106: "灯笼",
    107: "麻将",
    108: "麦克风",
    109: "幽灵",
    110: "蛋糕",
    111: "飞机",
    112: "钞票",
    113: "手枪",
    114: "茶",
    115: "眨眼",
    116: "泪奔",
    117: "无奈",
    118: "卖萌",
    119: "小纠结",
    120: "喷血",
    121: "斜眼笑",
    122: "doge",
    123: "惊喜",
    124: "骚扰",
    125: "笑哭",
    126: "我最美",
    127: "托腮",
    128: "点赞",
    129: "托腮",
    178: "狗头",
    179: "汪汪",
    180: "太阳",
    181: "月亮",
    182: "点赞",
    183: "踩",
    184: "握手",
    185: "胜利",
    212: "托腮",
    262: "面壁",
    264: "脑阔疼",
    267: "沧桑",
    277: "抱抱",
    324: "妙啊",
    326: "嗑瓜子",
}


_EXTRA_FACE_MAP: dict[str, str] = {}
_EXTRA_LOADED_AT: float = 0.0
_CATALOG_CANDIDATES = (
    r"D:\app1\SnowLuma\data\sys-face-catalog.json",
)


def _load_snowluma_catalog() -> dict[str, str]:
    """从 SnowLuma 的系统表情表补一份「新表情 / 超级表情」的 id→名字。

    QQ 的 1.1.x 新增了一大批超级表情（id 300+、400+），插件自带的表只到 326。
    SnowLuma（协议端）自己维护着一份 sys-face-catalog.json（qSid → qDes），
    直接借用它，就不用自己维护映射表了。
    """
    import json
    import os

    global _EXTRA_FACE_MAP, _EXTRA_LOADED_AT

    paths = list(_CATALOG_CANDIDATES)
    env_dir = os.environ.get("SNOWLUMA_DATA_DIR")
    if env_dir:
        paths.insert(0, os.path.join(env_dir, "sys-face-catalog.json"))
    paths.append(os.path.join(os.environ.get("APPDATA", ""), "SnowLuma", "data", "sys-face-catalog.json"))

    for path in paths:
        try:
            if not path or not os.path.isfile(path):
                continue
            if _EXTRA_FACE_MAP and os.path.getmtime(path) <= _EXTRA_LOADED_AT:
                return _EXTRA_FACE_MAP
            data = json.load(open(path, encoding="utf-8"))
            emojis = [
                e
                for pack in (data.get("packs") or [])
                for e in (pack.get("emojis") or [])
            ]

            # 第一遍：qSid 是 OneBot Face.id 的主键，必须先占位、之后不许被别的 id 空间覆盖。
            # 四个 id 空间混着塞会撞键：超级表情的 aniStickerId 和经典表情的 qSid 同号，
            # 实测 294 个数字型 qSid 里有 70 个会被抢（1 号「撇嘴」被「打call」抢、
            # 78 号「握手」被「不是吧」抢），所以 qSid 必须第一优先级进来。
            mapping: dict[str, str] = {}
            for emoji in emojis:
                name = str(emoji.get("qDes") or "").lstrip("/")
                k = str(emoji.get("qSid") or "").strip()
                if name and k and k not in mapping:
                    mapping[k] = name

            # 第二遍：emCode / aniStickerId / qCid 只补 qSid 里没有的键，
            # 保证「qSid 有名字就不会被其它字段改写」。
            for emoji in emojis:
                name = str(emoji.get("qDes") or "").lstrip("/")
                if not name:
                    continue
                for key in (
                    emoji.get("emCode"),
                    emoji.get("aniStickerId"),
                    emoji.get("qCid"),
                ):
                    k = str(key or "").strip()
                    if k and k not in mapping:
                        mapping[k] = name
            if mapping:
                _EXTRA_FACE_MAP = mapping
                _EXTRA_LOADED_AT = os.path.getmtime(path)
                return mapping
        except Exception:
            continue
    return _EXTRA_FACE_MAP


def _default_override_path() -> str:
    """覆盖表放在**本插件自己的** plugin_data 下（不写别人的目录）。"""

    import os
    from pathlib import Path

    try:
        plugin_dir = Path(__file__).resolve().parent           # .../data/plugins/<本插件>
        data_root = plugin_dir.parents[1]                      # .../data
        target = data_root / "plugin_data" / plugin_dir.name
        target.mkdir(parents=True, exist_ok=True)
        return str(target / "face_overrides.json")
    except Exception:  # noqa: BLE001
        return os.path.join(
            os.environ.get("TEMP", "."), "av_sense_face_overrides.json"
        )


_OVERRIDE_PATH = _default_override_path()
_OVERRIDE_MAP: dict[str, str] = {}
_OVERRIDE_MTIME: float = 0.0


def _load_overrides() -> dict[str, str]:
    """人工补的表：QQ 里点开表情会「朗读」出名字，用户听得到的那些名字补到这里。

    格式：{"491": "自信学霸", "492": "..."}；文件改了会自动重新读。
    """
    import json
    import os

    global _OVERRIDE_MAP, _OVERRIDE_MTIME
    try:
        if not os.path.isfile(_OVERRIDE_PATH):
            return _OVERRIDE_MAP
        mtime = os.path.getmtime(_OVERRIDE_PATH)
        if _OVERRIDE_MAP and mtime <= _OVERRIDE_MTIME:
            return _OVERRIDE_MAP
        raw = json.load(open(_OVERRIDE_PATH, encoding="utf-8"))
        _OVERRIDE_MAP = {str(k): str(v) for k, v in raw.items() if str(v).strip()}
        _OVERRIDE_MTIME = mtime
    except Exception:
        pass
    return _OVERRIDE_MAP


def add_override(face_id: int | str, name: str) -> bool:
    """运行时补一条映射并落盘。"""
    import json

    global _OVERRIDE_MTIME
    name = str(name or "").strip()
    if not name:
        return False
    mapping = dict(_load_overrides())
    mapping[str(face_id)] = name
    try:
        import os

        os.makedirs(os.path.dirname(_OVERRIDE_PATH) or ".", exist_ok=True)
        with open(_OVERRIDE_PATH, "w", encoding="utf-8") as fh:
            json.dump(mapping, fh, ensure_ascii=False, indent=2)
        _OVERRIDE_MTIME = 0.0
        _load_overrides()
        return True
    except Exception:
        return False


def get_face_name(face_id: int | str) -> str:
    """把 QQ 表情 id 翻译成中文（自带表 → SnowLuma 表情目录 → 占位符）。"""
    try:
        fid = int(face_id)
    except (TypeError, ValueError):
        return str(face_id)
    # 优先级：人工表 → 协议端官方目录 → 插件自带老表 → 占位符。
    # 自带的 QQ_FACE_MAP 是很多年前的表，跟现在的 QQ 对不上：
    # 比如它把 181 当「月亮」，而官方现在 181 是「戳一戳」——所以官方目录必须排前面。
    manual = _load_overrides().get(str(fid))
    if manual:
        return manual
    official = _load_snowluma_catalog().get(str(fid))
    if official:
        return official
    legacy = QQ_FACE_MAP.get(fid)
    if legacy:
        return legacy
    return f"表情_{fid}"
