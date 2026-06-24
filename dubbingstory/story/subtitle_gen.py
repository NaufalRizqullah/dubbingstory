"""
dubbingstory.story.subtitle_gen — SRT subtitle file generator

Generates .srt subtitle files from narration segments with proper timing.
"""

import os


def _format_srt_time(seconds: float) -> str:
    """
    Format seconds into SRT timestamp: HH:MM:SS,mmm

    Parameters
    ----------
    seconds : float
        Time in seconds.

    Returns
    -------
    str
        SRT-formatted timestamp.
    """
    seconds = max(0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(
    narration_segments: list[dict],
    output_path: str,
    max_chars_per_line: int = 42,
) -> str:
    """
    Generate an SRT subtitle file from narration segments.

    Parameters
    ----------
    narration_segments : list[dict]
        Narration segments with timing. Each dict must have:
        - scene_id: str
        - text: str
        - start_time: float
        - end_time: float
    output_path : str
        Path to save the .srt file.
    max_chars_per_line : int
        Maximum characters per subtitle line for readability.

    Returns
    -------
    str
        Path to the generated .srt file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    srt_entries = []
    index = 1

    for seg in narration_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        start = seg.get("start_time", 0)
        end = seg.get("end_time", start + seg.get("estimated_duration", 5.0))

        # Split long text into multiple subtitle blocks
        blocks = _split_text_for_subtitle(text, max_chars_per_line)

        if len(blocks) == 1:
            # Single block
            srt_entries.append(
                f"{index}\n"
                f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n"
                f"{blocks[0]}\n"
            )
            index += 1
        else:
            # Multiple blocks — distribute timing
            block_duration = (end - start) / len(blocks)
            for i, block in enumerate(blocks):
                block_start = start + (i * block_duration)
                block_end = block_start + block_duration

                srt_entries.append(
                    f"{index}\n"
                    f"{_format_srt_time(block_start)} --> {_format_srt_time(block_end)}\n"
                    f"{block}\n"
                )
                index += 1

    # Write SRT file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_entries))

    return output_path


def _split_text_for_subtitle(text: str, max_chars: int = 42) -> list[str]:
    """
    Split text into subtitle-friendly blocks.

    Each block is max 2 lines, each line max_chars characters.
    Splits at sentence boundaries preferentially, then word boundaries.
    """
    if len(text) <= max_chars * 2:
        # Short enough for one block (up to 2 lines)
        if len(text) <= max_chars:
            return [text]
        # Split into 2 lines
        return [_wrap_text(text, max_chars)]

    # Split at sentence boundaries first
    import re
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)

    blocks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars * 2:
            current = f"{current} {sentence}".strip() if current else sentence
        else:
            if current:
                blocks.append(_wrap_text(current, max_chars))
            current = sentence

    if current:
        blocks.append(_wrap_text(current, max_chars))

    return blocks if blocks else [text]


def _wrap_text(text: str, max_chars: int) -> str:
    """Wrap text into max 2 lines for subtitle display."""
    if len(text) <= max_chars:
        return text

    # Find best split point near the middle
    mid = len(text) // 2
    best_split = mid

    # Look for space near middle
    for offset in range(0, mid):
        if mid + offset < len(text) and text[mid + offset] == " ":
            best_split = mid + offset
            break
        if mid - offset >= 0 and text[mid - offset] == " ":
            best_split = mid - offset
            break

    line1 = text[:best_split].strip()
    line2 = text[best_split:].strip()

    return f"{line1}\n{line2}"
