# src/videorag/retrieval/translator.py — Multilingual Visual Semantic Mapper
from __future__ import annotations
import re

# Conversational prefixes and fillers to strip
STOP_PREFIXES = [
    r"^اعرض\s+لي\s+(لقطه|مقطع|فيديو)?\s*",
    r"^ابحث\s+عن\s*",
    r"^اريد\s+(لقطه|مقطع|فيديو|صوره)?\s*",
    r"^أريد\s+(لقطه|مقطع|فيديو|صوره)?\s*",
    r"^جيب\s+لي\s*",
    r"^فين\s+(اللقطه|المقطع|الصوره|الفيديو)?\s*",
    r"^أظهر\s+لي\s*",
    r"^اظهر\s+لي\s*",
    r"^لقطة\s+(تظهر|فيها|مع)?\s*",
    r"^مقطع\s+(يظهر|فيه|مع)?\s*",
    r"^فيديو\s+(يظهر|فيه|مع)?\s*",
    r"^لحظة\s+(ظهور|عرض)?\s*",
    r"^أين\s+(يظهر|تظهر)?\s*",
    r"^اين\s+(يظهر|تظهر)?\s*",
]

# Comprehensive visual concept mapping (Arabic -> English)
AR_EN_MAP = {
    # Winter & Skiing
    "متزلجين": "skiers",
    "متزلج": "skier",
    "تزلج": "skiing snow",
    "التزلج": "skiing",
    "ثلج": "snow",
    "الثلج": "snow",
    "جليد": "ice",
    "شتاء": "winter",

    # Nature & Landscapes
    "جبل": "mountain",
    "جبال": "mountains",
    "الجبال": "mountains",
    "قمة": "peak",
    "قمم": "mountain peaks",
    "بحر": "sea ocean",
    "البحر": "sea ocean",
    "محيط": "ocean",
    "شاطئ": "beach",
    "الشاطئ": "beach",
    "ساحل": "coast coastline",
    "أمواج": "ocean waves",
    "ماء": "water",
    "غيوم": "clouds",
    "الغيوم": "clouds",
    "سحاب": "clouds",
    "ضباب": "fog misty",
    "سماء": "sky",
    "غابة": "forest",
    "الغابة": "forest trees",
    "شجر": "trees",
    "أشجار": "trees",
    "طبيعة": "nature landscape",
    "صحراء": "desert",
    "الصحراء": "desert sand",
    "رمل": "sand",
    "رمال": "sand dunes",
    "كثبان": "sand dunes",
    "نهر": "river",
    "شلال": "waterfall",
    "بحيرة": "lake",
    "شروق": "sunrise",
    "غروب": "sunset",

    # Architecture & Cities
    "مدينة": "city skyline",
    "المدينة": "city",
    "مبنى": "building",
    "مباني": "buildings architecture",
    "برج": "tower skyscraper",
    "ناطحة سحاب": "skyscraper",
    "ناطحات": "skyscrapers",
    "شارع": "street road",
    "شوارع": "city streets",
    "بيوت": "houses village",
    "قرية": "village",
    "قلعة": "castle fortress",
    "قصر": "palace",
    "كنيسة": "church cathedral",
    "الكنيسة": "church cathedral",
    "معبد": "temple",
    "مسجد": "mosque",
    "جامع": "mosque",

    # Vehicles & Transportation
    "طائرة": "airplane flying aerial",
    "طائرات": "airplanes",
    "سيارة": "car",
    "سيارات": "cars traffic",
    "قطار": "train railway",
    "سفينة": "ship boat",
    "قارب": "boat sailing",
    "يخت": "yacht",
    "دراجة": "bicycle motorcycle",

    # Tech & Computing
    "حاسوب": "computer",
    "كمبيوتر": "computer screen",
    "شاشة": "computer screen display",
    "سطر الأوامر": "terminal command line interface",
    "تيرمينال": "terminal CLI window",
    "سطر أوامر": "command line interface",
    "أوامر": "commands terminal",
    "كود": "programming code",
    "برمجة": "code coding",
    "سكريبت": "script terminal",
    "بايثون": "python programming",
    "شاشة سوداء": "dark terminal screen",

    # People & Actions
    "شخص": "person",
    "رجل": "man",
    "رجال": "men",
    "امرأة": "woman",
    "نساء": "women",
    "أطفال": "children kids",
    "طفل": "child kid",
    "ناس": "people crowd",
    "حشود": "crowd",
    "ركض": "running",
    "جري": "running athlete",
    "مشي": "walking",
    "سباحة": "swimming",
    "طيران": "flying aerial view",
    "نظارة": "glasses",
    "قبعة": "hat",
    "خوذة": "helmet",

    # Aerial & Perspectives
    "جو": "aerial aerial view",
    "الجو": "aerial drone view",
    "من الجو": "aerial view drone",
    "طائرة مسيرة": "drone view",
    "درون": "drone shot",
}


def is_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def adapt_visual_query(query: str) -> str:
    """
    Transforms a natural language query (especially in Arabic) into
    an optimized English visual concept description for CLIP ViT-B/32.

    If query is already English, returns clean query.
    """
    clean = query.strip()

    # Strip conversational prefixes
    for pat in STOP_PREFIXES:
        clean = re.sub(pat, "", clean, flags=re.IGNORECASE).strip()

    if not is_arabic(clean):
        return clean if clean else query

    # Translate known multi-word phrases first
    translated_tokens: list[str] = []
    text_remaining = clean

    # Sort keys by length descending to match multi-word phrases first
    sorted_phrases = sorted(AR_EN_MAP.keys(), key=lambda k: len(k.split()), reverse=True)

    for phrase in sorted_phrases:
        if phrase in text_remaining:
            translated_tokens.append(AR_EN_MAP[phrase])
            text_remaining = text_remaining.replace(phrase, " ")

    # For remaining individual words, check map
    words = re.findall(r"[\u0600-\u06FF]+", text_remaining)
    for w in words:
        if w in AR_EN_MAP:
            translated_tokens.append(AR_EN_MAP[w])

    if translated_tokens:
        result = " ".join(dict.fromkeys(translated_tokens))  # preserve order, deduplicate
        return result

    # Fallback: if no dictionary match, return original clean string
    return clean
