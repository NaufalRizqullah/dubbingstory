"""SRT/ASS filter construction shared by standalone burn and final rendering."""

import os

from dubbingstory.render.audio_mix import _run_ffmpeg


def _escape_filter_path(path: str) -> str:
    """Escape both FFmpeg option parsing and filtergraph parsing, without a shell."""
    value = os.path.abspath(path).replace("\\", "/")
    value = "".join("\\" + c if c in "\\':" else c for c in value)
    return "".join("\\" + c if c in "\\'[],;" else c for c in value)


def _ass_color(value: str) -> str:
    named = {"white": "FFFFFF", "black": "000000", "yellow": "FFFF00",
             "red": "FF0000", "green": "00FF00", "blue": "0000FF"}
    value = named.get(value.lower(), value.lstrip("#"))
    if len(value) != 6 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError("Subtitle colors must be named basic colors or #RRGGBB")
    return f"&H00{value[4:6]}{value[2:4]}{value[0:2]}"


def build_subtitle_filter(
    subtitle_path: str, *, font_size: int = 24, font_color: str = "white",
    outline_color: str = "black", outline_width: float = 2,
    position: str = "bottom", font_family: str = "Arial", margin_v: int = 30,
    fonts_dir: str | None = None,
    original_size: tuple[int, int] | None = None,
) -> str:
    if not os.path.isfile(subtitle_path):
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")
    expression = f"subtitles=filename={_escape_filter_path(subtitle_path)}"
    if fonts_dir:
        expression += f":fontsdir={_escape_filter_path(fonts_dir)}"
    if original_size:
        width, height = original_size
        if width <= 0 or height <= 0:
            raise ValueError("Subtitle original_size must be positive display dimensions")
        expression += f":original_size={int(width)}x{int(height)}"
    # ASS contains resolution-aware layout and animation. Never override it with
    # the classic SRT style, which would destroy its configured appearance.
    if os.path.splitext(subtitle_path)[1].lower() == ".ass":
        return expression
    if any(c in font_family for c in "'\\,:;[]\r\n"):
        raise ValueError("Invalid font family name")
    alignment = {"bottom": 2, "center": 5, "top": 8}.get(position)
    if alignment is None:
        raise ValueError("Subtitle position must be bottom, center or top")
    style = (f"FontName={font_family},FontSize={int(font_size)},"
             f"PrimaryColour={_ass_color(font_color)},OutlineColour={_ass_color(outline_color)},"
             f"BorderStyle=1,Outline={float(outline_width)},MarginV={int(margin_v)},Alignment={alignment}")
    return expression + f":force_style='{style}'"


def burn_subtitles(
    video_path: str, subtitle_path: str, output_path: str, font_size: int = 24,
    font_color: str = "white", outline_color: str = "black", outline_width: int = 2,
    position: str = "bottom", font_family: str = "Arial", margin_v: int = 30,
    fonts_dir: str | None = None,
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    vf = build_subtitle_filter(subtitle_path, font_size=font_size, font_color=font_color,
                              outline_color=outline_color, outline_width=outline_width,
                              position=position, font_family=font_family,
                              margin_v=margin_v, fonts_dir=fonts_dir)
    _run_ffmpeg(["ffmpeg", "-y", "-i", video_path, "-vf", vf,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-fps_mode", "passthrough",
                 "-c:a", "copy", output_path], "subtitle burn (FFmpeg with libass required)")
    return output_path
