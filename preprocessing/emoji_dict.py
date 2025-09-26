#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emoji Dictionary for Text Cleaning
Mapping common emojis and crypto-related symbols to textual descriptions.

Author: Bitcoin Price Prediction Team
Date: 2025-09-20
"""

# Core mapping. Add more as needed for your dataset.
EMOJI_DICT = {
    # Faces - positive
    "😀": " smile ",
    "😃": " smile ",
    "😄": " smile ",
    "😁": " grin ",
    "😆": " laugh ",
    "😂": " laugh ",
    "🤣": " rofl ",
    "😊": " blush ",
    "🙂": " smile ",
    "😉": " wink ",
    "😍": " love ",
    "🥰": " love ",
    "😘": " kiss ",
    "😙": " kiss ",
    "😚": " kiss ",
    "😎": " cool ",

    # Faces - neutral/other
    "🤔": " thinking ",
    "🤨": " skeptical ",
    "😐": " neutral ",
    "😶": " speechless ",
    "😴": " sleepy ",
    "🤤": " drool ",

    # Faces - negative
    "🙁": " sad ",
    "☹️": " sad ",
    "😞": " disappointed ",
    "😟": " worried ",
    "😢": " cry ",
    "😭": " sob ",
    "😤": " frustrated ",
    "😠": " angry ",
    "😡": " angry ",
    "🤯": " mindblown ",
    "😱": " shocked ",
    "😨": " scared ",
    "😰": " anxious ",
    "😓": " sweat ",
    "🤬": " rage ",

    # Gestures
    "👍": " thumbs_up ",
    "👎": " thumbs_down ",
    "🙏": " pray ",
    "👏": " clap ",
    "🙌": " praise ",
    "💪": " strong ",
    "👌": " ok ",
    "✌️": " victory ",
    "🤝": " handshake ",
    "🫡": " salute ",

    # Hearts and emotions
    "❤": " heart ",
    "❤️": " heart ",
    "💙": " blue_heart ",
    "💚": " green_heart ",
    "💛": " yellow_heart ",
    "💜": " purple_heart ",
    "🖤": " black_heart ",
    "💔": " broken_heart ",
    "🔥": " fire ",
    "✨": " sparkles ",
    "⭐": " star ",
    "🌟": " star ",

    # Money and trading
    "💰": " money_bag ",
    "💸": " money_flying ",
    "💵": " dollar ",
    "💴": " yen ",
    "💶": " euro ",
    "💷": " pound ",
    "📈": " chart_up ",
    "📉": " chart_down ",
    "🟩": " green_square ",
    "🟥": " red_square ",

    # Crypto and symbols
    "₿": " bitcoin ",
    "#️⃣": " hashtag ",

    # Transport/memes
    "🚀": " rocket ",
    "🚨": " alert ",
    "🏎️": " lambo ",
    "🚗": " car ",

    # Time
    "⏳": " hourglass ",
    "⌛": " hourglass ",
    "⏰": " alarm ",
    "🕒": " clock ",

    # Up/Down/Arrows
    "⬆️": " up ",
    "⬇️": " down ",
    "➡️": " right ",
    "⬅️": " left ",
    "🔼": " up ",
    "🔽": " down ",

    # Objects common in tweets
    "🔔": " bell ",
    "💡": " idea ",
    "📢": " announce ",
    "📰": " news ",
    "📊": " chart ",
    "📉": " chart_down ",
    "📈": " chart_up ",

    # Weather/metaphors
    "🌙": " moon ",
    "🌕": " full_moon ",
    "🌑": " new_moon ",
    "☀️": " sun ",
    "🌧️": " rain ",
    "⛈️": " storm ",
    "🌪️": " tornado ",

    # Misc
    "🎉": " party ",
    "✅": " check ",
    "❌": " cross ",
    "⚠️": " warning ",
    "❗": " exclamation ",
    "❓": " question ",
}

# Aliases for common sequences and emojis that vary by platform
EMOJI_ALIASES = {
    "❤": "❤️",
}

__all__ = ["EMOJI_DICT", "EMOJI_ALIASES", "convert_emojis_to_text"]

def convert_emojis_to_text(text: str) -> str:
    """
    Convert emojis in a string to textual descriptions using EMOJI_DICT.
    Unknown emojis are left as-is.
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""

    # Normalize aliases
    for alias, canonical in EMOJI_ALIASES.items():
        if alias in text:
            text = text.replace(alias, canonical)

    for emoji, description in EMOJI_DICT.items():
        if emoji in text:
            text = text.replace(emoji, description)
    return text 