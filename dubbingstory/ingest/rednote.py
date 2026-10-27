"""Preserve public RedNote URLs while reusing yt-dlp's maintained XHS parser.

The upstream XiaoHongShu extractor matches only xiaohongshu.com. Rewriting a
RedNote link to that host can yield an error page even while the original public
RedNote page exposes its video. Extend URL matching only: no alternate API,
authentication, cookies, browser impersonation or extraction logic is added.
"""

from yt_dlp.extractor.xiaohongshu import XiaoHongShuIE


class RedNoteIE(XiaoHongShuIE):
    IE_NAME = "RedNote"
    IE_DESC = "RedNote public video pages"
    _VALID_URL = r"https?://www\.rednote\.com/(?:explore|discovery/item)/(?P<id>[\da-f]+)"
    _TESTS = []  # Upstream fixtures target the other host; local tests use mocks.
