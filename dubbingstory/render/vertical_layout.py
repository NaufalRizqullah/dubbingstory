"""Centered two-layer portrait composition, independent of audio and narration."""
from __future__ import annotations

import math


def layout_settings(cfg=None) -> dict:
    """Validate settings once for both CLI configuration and direct render calls."""
    layout = getattr(cfg, "render_vertical_layout", "pad")
    if layout not in {"pad", "layered"}:
        raise ValueError("render.vertical_layout must be pad or layered")
    result = {"layout": layout}
    for name, default, minimum, maximum in (
        ("foreground_zoom", 25, 0, 100),
        ("background_zoom", 100, 0, 100),
        ("background_blur", 50, 1, 100),
        ("background_grayscale", 1.0, 0, 1),
    ):
        try:
            value = float(getattr(cfg, "render_" + name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"render.{name} must be a number between {minimum} and {maximum}") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"render.{name} must be between {minimum} and {maximum}")
        result[name] = value
    return result


def layer_dimensions(metadata: dict, canvas_width: int, canvas_height: int, zoom: float) -> tuple[int, int]:
    """Interpolate from contain (0) to cover (100), preserving display aspect.

    Display dimensions already account for phone rotation and sample aspect
    ratio. The renderer's normal FFmpeg autorotation stays enabled. Each branch
    scales straight to its final square-pixel size; there is no intermediate
    full-frame resize. Even-pixel rounding supports yuv420p/H.264.
    """
    try:
        width, height = float(metadata["display_width"]), float(metadata["display_height"])
        zoom = float(zoom)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Layered rendering requires valid source display dimensions and zoom") from exc
    if (not all(math.isfinite(value) for value in (width, height, zoom, canvas_width, canvas_height))
            or min(width, height, canvas_width, canvas_height) <= 0 or not 0 <= zoom <= 100):
        raise ValueError("Invalid layered rendering dimensions or zoom (expected 0–100)")
    fit = min(canvas_width / width, canvas_height / height)
    fill = max(canvas_width / width, canvas_height / height)
    scale = fit + (fill - fit) * zoom / 100
    # Avoid an extra pair of pixels from floating-point noise at an exact edge.
    return tuple(max(2, 2 * math.ceil(value * scale / 2 - 1e-9)) for value in (width, height))


def build_layered_video_filter(
    video_selector: str, metadata: dict, width: int, height: int, cfg=None,
) -> str:
    """Return a video-only graph ending in [vertical_composite].

    Both layers originate from the same timestamped frames. The background
    drives overlay timing, including VFR; no synthetic color source, fps filter,
    second input or audio branch is introduced. Captions belong after this graph.
    """
    settings = layout_settings(cfg)
    foreground = layer_dimensions(metadata, width, height, settings["foreground_zoom"])
    background = layer_dimensions(metadata, width, height, settings["background_zoom"])

    def scaled_crop(size):
        scaled_w, scaled_h = size
        return (
            f"scale={scaled_w}:{scaled_h}:flags=lanczos,setsar=1,"
            f"crop={min(width, scaled_w)}:{min(height, scaled_h)}:(iw-ow)/2:(ih-oh)/2"
        )

    # Blur is sigma in canvas pixels. Desaturation is explicit: 0 preserves
    # color, 1 removes all saturation. Neither affects the foreground branch.
    sigma = settings["background_blur"]
    saturation = 1 - settings["background_grayscale"]
    return (
        f"[{video_selector}]setpts=PTS-STARTPTS,split=2[vertical_bg_in][vertical_fg_in];"
        f"[vertical_bg_in]{scaled_crop(background)},"
        f"hue=s={saturation:.6f},gblur=sigma={sigma:.6f}:steps=2,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black[vertical_bg];"
        f"[vertical_fg_in]{scaled_crop(foreground)}[vertical_fg];"
        "[vertical_bg][vertical_fg]overlay=x=(W-w)/2:y=(H-h)/2:"
        "eof_action=pass:repeatlast=0:shortest=0:format=auto[vertical_composite]"
    )
