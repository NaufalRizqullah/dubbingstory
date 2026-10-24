"""ASS captions driven exclusively by measured TTS word timestamps."""

from __future__ import annotations

import json
import math
import os
import re

from dubbingstory.tts.word_alignment import validate_word_timings


def _escape_ass(text: str) -> str:
    # ASS has no universally safe quoted-text syntax: doubling a backslash can
    # still leave a following \N active in libass. Use visible fullwidth forms
    # for reserved literal characters so narration cannot inject override tags.
    return text.translate(str.maketrans({"\\": "＼", "{": "｛", "}": "｝", "\r": " ", "\n": " "}))


def _color(value: str, alpha: bool = True) -> str:
    value = str(value).lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        raise ValueError("Subtitle colors must be six-digit RGB hex values, e.g. #FFFFFF.")
    bgr = value[4:6] + value[2:4] + value[:2]
    return f"&H{'00' if alpha else ''}{bgr.upper()}{'' if alpha else '&'}"


def _time(centiseconds: int) -> str:
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{fraction:02d}"


def _event(start: float, end: float, text: str) -> str:
    # Do not manufacture a tick for collapsed intervals: that could overlap
    # the next word and display two active words at once.
    first = max(0, round(start * 100))
    last = round(end * 100)
    if last <= first:
        raise ValueError("A word interval cannot be represented at ASS 10-ms precision; review or regenerate the word alignment.")
    return f"Dialogue: 0,{_time(first)},{_time(last)},Caption,,0,0,0,,{text}"


def _load_font(name: str, font_path: str | None, size: int):
    from PIL import ImageFont

    candidates = [font_path] if font_path else [name, name + ".ttf", name.replace(" ", "") + ".ttf"]
    if not font_path and name == "DejaVu Sans":
        candidates.append("DejaVuSans.ttf")
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, ValueError):
            continue
    raise ValueError(
        f"Caption font '{name}' could not be measured. Set subtitle.font_path to its TTF/OTF file and subtitle.font_name to the font's installed family."
    )


def caption_font_directory(cfg) -> str:
    """Directory to register in libass, using the same font measured by Pillow."""
    font = _load_font(
        str(getattr(cfg, "subtitle_font_name", "DejaVu Sans")),
        getattr(cfg, "subtitle_font_path", None), 16,
    )
    font_path = getattr(font, "path", None)
    if not font_path or not os.path.isfile(font_path):
        raise ValueError("The measured caption font has no readable font file. Set subtitle.font_path to its TTF/OTF file.")
    return os.path.dirname(os.path.abspath(font_path))


def _line_width(font, text: str) -> float:
    box = font.getbbox(text)
    return float(box[2] - box[0])


def _layout(words: list[dict], font, width: float) -> list[list[int]]:
    lines: list[list[int]] = [[]]
    for index, word in enumerate(words):
        candidate = lines[-1] + [index]
        label = " ".join(_escape_ass(words[i]["text"]) for i in candidate)
        if lines[-1] and _line_width(font, label) > width:
            lines.append([index])
        else:
            lines[-1] = candidate
    return lines


def _phrases(words: list[dict], font, width: float, max_words: int) -> list[list[dict]]:
    phrases, current = [], []
    for word in words:
        candidate = current + [word]
        split = bool(current) and (
            len(candidate) > max_words
            or word["start"] - current[-1]["end"] > 0.65
            or word.get("scene_id") != current[-1].get("scene_id")
            or len(_layout(candidate, font, width)) > 2
        )
        if split:
            phrases.append(current)
            current = [word]
        else:
            current = candidate
        if re.search(r"[.!?]$", word["text"]):
            phrases.append(current)
            current = []
    if current:
        phrases.append(current)
    return phrases


def _animation_tags(
    animation: str, duration: float, pop_scale: float,
    elapsed: float = 0.0, interval_duration: float | None = None,
) -> str:
    """Evaluate one continuous phrase animation on an interval's local clock.

    Karaoke changes color through separate events. Their scale/opacity must
    continue from the phrase clock, even when a word ends inside the pop ramp.
    """
    total = max(1.0, duration * 1000)
    first = max(0.0, elapsed * 1000)
    last = min(total, first + (interval_duration * 1000 if interval_duration is not None else total))
    ramp = min(90.0, max(1.0, total / 3))
    if animation in {"pop", "scale"}:
        initial = pop_scale * 100 if animation == "pop" else 85.0

        def value(at):
            return initial + (100.0 - initial) * min(1.0, at / ramp)

        def tags(at):
            scale = value(at)
            return rf"\fscx{scale:.3f}\fscy{scale:.3f}"

        result = tags(first)
        if first < ramp:
            finish = min(last, ramp)
            result += rf"\t(0,{max(1, round(finish - first))},{tags(finish)})"
        return result
    if animation == "fade":
        fade_in, fade_out = min(40.0, total / 4), min(30.0, total / 4)

        def tags(at):
            opacity = min(1.0, at / fade_in, (total - at) / fade_out)
            alpha = round(255 * (1 - max(0.0, opacity)))
            return rf"\alpha&H{alpha:02X}&"

        result = tags(first)
        previous = first
        for at in sorted({point for point in (fade_in, total - fade_out, last) if first < point <= last}):
            result += rf"\t({round(previous - first)},{max(1, round(at - first))},{tags(at)})"
            previous = at
        return result
    return ""


def generate_ass(
    payload_or_json_path: dict | str, output_path: str, width: int, height: int,
    cfg, style: str | None = None,
) -> str:
    """Generate word-by-word or active-word-only karaoke ASS.

    Font size is specified at 1080-pixel canvas width and scaled proportionally.
    Safe margins are fractions of display width/height. Karaoke keeps the same
    phrase layout across active-word changes, including neutral text in gaps.
    The renderer must validate the artifact's audio/text hashes before calling
    this function; generation itself can be used independently for previews.
    """
    if isinstance(payload_or_json_path, (str, os.PathLike)):
        with open(payload_or_json_path, encoding="utf-8") as source:
            payload = json.load(source)
    else:
        payload = payload_or_json_path
    words = validate_word_timings(payload.get("words", []), text=payload.get("text"))
    quantized = []
    previous_end_tick = 0
    for word in words:
        start_tick, end_tick = round(word["start"] * 100), round(word["end"] * 100)
        if end_tick <= start_tick or start_tick < previous_end_tick:
            raise ValueError("Word timing collapses or overlaps at ASS 10-ms precision; review or regenerate alignment instead of guessing caption times.")
        quantized.append({**word, "start": start_tick / 100, "end": end_tick / 100})
        previous_end_tick = end_tick
    words = quantized
    style = style or getattr(cfg, "subtitle_style", "word")
    if style not in {"word", "karaoke"}:
        raise ValueError("Animated subtitle style must be word or karaoke.")
    if width <= 0 or height <= 0:
        raise ValueError("Subtitle display dimensions must be positive.")
    font_name = str(getattr(cfg, "subtitle_font_name", "DejaVu Sans"))
    if any(char in font_name for char in ",\r\n{}"):
        raise ValueError("Subtitle font name contains ASS control characters.")
    reference_size = float(getattr(cfg, "subtitle_font_size", 64))
    if not math.isfinite(reference_size) or reference_size <= 0:
        raise ValueError("Subtitle font size must be positive and finite.")
    size = max(1, round(reference_size * width / 1080))
    font = _load_font(font_name, getattr(cfg, "subtitle_font_path", None), size)
    # The font file is authoritative; a mismatched configured family must not
    # make libass select a different font from the one used for layout.
    measured_family, measured_style = font.getname()
    font_name = str(measured_family)
    bold = -1 if "bold" in str(measured_style).casefold() else 0
    italic = -1 if any(part in str(measured_style).casefold() for part in ("italic", "oblique")) else 0
    if any(char in font_name for char in ",\r\n{}"):
        raise ValueError("The measured font's family name cannot be represented in an ASS style.")
    margin_x = float(getattr(cfg, "subtitle_safe_margin_x", 0.08))
    margin_y = float(getattr(cfg, "subtitle_safe_margin_y", 0.14))
    if not (0 <= margin_x < 0.45 and 0 <= margin_y < 0.45):
        raise ValueError("Subtitle safe margins must be fractions between 0 and 0.45.")
    outline = float(getattr(cfg, "subtitle_outline", 3)) * width / 1080
    if not math.isfinite(outline) or outline < 0:
        raise ValueError("Subtitle outline must be nonnegative.")
    animation = str(getattr(cfg, "subtitle_animation", "pop"))
    if animation not in {"none", "pop", "fade", "scale"}:
        raise ValueError("Subtitle animation must be none, pop, fade, or scale.")
    pop_scale = float(getattr(cfg, "subtitle_pop_scale", 1.12))
    if not math.isfinite(pop_scale) or not 1.0 <= pop_scale <= 1.5:
        raise ValueError("subtitle.pop_scale must be between 1.0 and 1.5.")
    maximum_scale = pop_scale if animation == "pop" else 1.0
    usable_width = width * (1 - 2 * margin_x) / maximum_scale - 2 * outline
    usable_height = height * (1 - 2 * margin_y) / maximum_scale - 2 * outline
    if usable_width <= 0 or usable_height <= 0:
        raise ValueError("Subtitle safe area is too small for the configured outline.")
    position = str(getattr(cfg, "subtitle_position", "bottom"))
    alignments = {"bottom": 2, "center": 5, "top": 8}
    if position not in alignments:
        raise ValueError("Subtitle position must be bottom, center, or top.")
    alignment = alignments[position]
    y = height / 2 if position == "center" else (
        height * margin_y + outline * maximum_scale if position == "top" else height * (1 - margin_y) - outline * maximum_scale
    )
    base = _color(getattr(cfg, "subtitle_font_color", "#FFFFFF"))
    active = _color(getattr(cfg, "subtitle_active_color", "#FFD54F"), alpha=False)
    neutral = _color(getattr(cfg, "subtitle_font_color", "#FFFFFF"), alpha=False)
    border = _color(getattr(cfg, "subtitle_outline_color", "#000000"))
    max_words = int(getattr(cfg, "subtitle_max_words", 5))
    if max_words < 1:
        raise ValueError("subtitle.max_words must be positive.")
    phrases = [[word] for word in words] if style == "word" else _phrases(words, font, usable_width, max_words)
    events = []
    for phrase in phrases:
        lines = _layout(phrase, font, usable_width)
        widest = max(_line_width(font, " ".join(_escape_ass(phrase[i]["text"]) for i in line)) for line in lines)
        ascent, descent = font.getmetrics()
        block_height = (ascent + descent) * len(lines)
        fit_scale = min(1.0, usable_width / max(widest, 1), usable_height / max(block_height, 1))
        if fit_scale < 0.5:
            raise ValueError("A caption cannot fit the configured safe area legibly. Reduce font size or widen the safe area.")
        fitted_size = max(1, math.floor(size * fit_scale))
        prefix = rf"\an{alignment}\pos({width / 2:.1f},{y:.1f})\q2\fs{fitted_size}"

        def phrase_text(active_index: int | None) -> str:
            rendered_lines = []
            for line in lines:
                rendered_words = []
                for index in line:
                    label = _escape_ass(phrase[index]["text"])
                    if style == "karaoke":
                        color = active if index == active_index else neutral
                        label = "{" + rf"\c{color}" + "}" + label
                    rendered_words.append(label)
                rendered_lines.append(" ".join(rendered_words))
            return r"\N".join(rendered_lines)

        if style == "word":
            word = phrase[0]
            tags = prefix + _animation_tags(animation, word["end"] - word["start"], pop_scale)
            events.append(_event(word["start"], word["end"], "{" + tags + "}" + phrase_text(None)))
        else:
            # Every interval carries the same phrase layout. Only its active
            # word color changes, using the original phrase's animation clock.
            phrase_start, phrase_end = phrase[0]["start"], phrase[-1]["end"]
            # Per-interval events retain neutral text through actual silences.
            boundaries = sorted({word[key] for word in phrase for key in ("start", "end")})
            for start, end in zip(boundaries, boundaries[1:]):
                midpoint = (start + end) / 2
                index = next((i for i, word in enumerate(phrase) if word["start"] <= midpoint < word["end"]), None)
                interval_tags = prefix + _animation_tags(
                    animation, phrase_end - phrase_start, pop_scale,
                    elapsed=start - phrase_start, interval_duration=end - start,
                )
                events.append(_event(start, end, "{" + interval_tags + "}" + phrase_text(index)))

    header = f"""[Script Info]
Title: DubbingStory measured word captions
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font_name},{size},{base},{base},{border},&H00000000,{bold},{italic},0,0,100,100,0,0,1,{outline:.2f},0,{alignment},{round(width * margin_x)},{round(width * margin_x)},{round(height * margin_y)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output:
        output.write(header + "\n".join(events) + "\n")
    return output_path
