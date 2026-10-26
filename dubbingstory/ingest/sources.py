"""Small source adapters and bounded share-link resolution for public videos.

Query strings are retained only in memory for the extractor. Persisted identities
and user-facing diagnostics never include share tokens or signed media URLs.
"""

from dataclasses import dataclass, field
import hashlib
import ipaddress
import re
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class SourceIngestError(RuntimeError):
    """Actionable ingest failure with a stable machine-readable category."""

    def __init__(self, category: str, message: str):
        self.category = category
        super().__init__(f"[{category}] {message}")


_HOSTS = {
    "youtube.com": "youtube", "www.youtube.com": "youtube",
    "m.youtube.com": "youtube", "music.youtube.com": "youtube",
    "youtu.be": "youtube",
    "douyin.com": "douyin", "www.douyin.com": "douyin",
    "v.douyin.com": "douyin", "www.iesdouyin.com": "douyin",
    "iesdouyin.com": "douyin",
    "xiaohongshu.com": "xiaohongshu", "www.xiaohongshu.com": "xiaohongshu",
    "m.xiaohongshu.com": "xiaohongshu", "rednote.com": "xiaohongshu",
    "www.rednote.com": "xiaohongshu", "xhslink.com": "xiaohongshu",
    "www.xhslink.com": "xiaohongshu",
    "tiktok.com": "tiktok", "www.tiktok.com": "tiktok",
    "m.tiktok.com": "tiktok", "vm.tiktok.com": "tiktok",
    "vt.tiktok.com": "tiktok",
}
_SHORT_HOSTS = {
    "v.douyin.com", "xhslink.com", "www.xhslink.com",
    "vm.tiktok.com", "vt.tiktok.com",
}
_EXTRACTOR_KEYS = {
    "youtube": "Youtube", "douyin": "Douyin",
    "xiaohongshu": "XiaoHongShu", "tiktok": "TikTok",
}


@dataclass(frozen=True)
class SourceURL:
    provider: str
    url: str = field(repr=False)  # Temporary share token; never log/serialize.
    canonical_url: str
    video_id: str | None
    is_short_link: bool = False

    @property
    def identity(self) -> str:
        value = self.video_id or hashlib.sha256(self.canonical_url.encode()).hexdigest()[:20]
        return f"{self.provider}:{value}"

    @property
    def extractor_key(self) -> str:
        return _EXTRACTOR_KEYS[self.provider]


def _extract_url(value: str) -> str:
    """Accept a URL or a platform share message containing exactly one URL."""
    candidates = re.findall(r"https?://[^\s<>\"\u300a\u300b]+", str(value).strip(), re.I)
    if len(candidates) != 1:
        raise SourceIngestError("unsupported_url", "Supply exactly one public video URL or share link.")
    return candidates[0].rstrip(".,;!?)]}\u3002\uff0c\uff1b\uff01\uff09")


def _parse_supported_url(value: str):
    try:
        candidate = _extract_url(value)
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        raise SourceIngestError("unsupported_url", "The video URL is malformed.") from None
    if (parsed.scheme.lower() not in {"http", "https"} or parsed.username or parsed.password
            or port not in {None, 80, 443} or "\\" in parsed.netloc
            or any(ord(ch) < 32 for ch in candidate)):
        raise SourceIngestError("unsupported_url", "Only public HTTP(S) video URLs without embedded credentials are supported.")
    host = (parsed.hostname or "").lower()
    if host not in _HOSTS:
        raise SourceIngestError("unsupported_url", "Supported URL sources are YouTube, TikTok, Douyin, and Xiaohongshu/RedNote.")
    return parsed, host


def normalize_source_url(value: str) -> SourceURL:
    """Normalize known video URLs without network access; reject profiles/playlists."""
    parsed, host = _parse_supported_url(value)
    provider = _HOSTS[host]
    path = parsed.path.rstrip("/") or "/"
    query = parsed.query
    if host in _SHORT_HOSTS:
        if path == "/":
            raise SourceIngestError("unsupported_url", "The share link has no item identifier.")
        canonical = urlunsplit(("https", host, path, "", ""))
        return SourceURL(provider, urlunsplit(("https", host, path, query, "")), canonical, None, True)
    if provider == "youtube":
        if host == "youtu.be":
            video_id = path.lstrip("/")
        elif path == "/watch":
            video_id = (parse_qs(query).get("v") or [""])[0]
        else:
            match = re.fullmatch(r"/(?:shorts|embed|live)/([\w-]+)", path)
            video_id = match.group(1) if match else ""
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise SourceIngestError("unsupported_url", "Provide a single YouTube video, not a channel or playlist.")
        canonical = f"https://www.youtube.com/watch?v={video_id}"
        return SourceURL(provider, canonical, canonical, video_id)
    if provider == "douyin":
        match = re.fullmatch(r"/(?:video|share/video)/(\d+)", path)
        if not match:
            raise SourceIngestError("unsupported_url", "Provide a Douyin video URL or v.douyin.com share link.")
        video_id = match.group(1)
        canonical = f"https://www.douyin.com/video/{video_id}"
    elif provider == "tiktok":
        match = re.fullmatch(r"/(?:@[^/]+/video|v)/(\d+)", path)
        if not match:
            raise SourceIngestError("unsupported_url", "Provide a TikTok video URL or vm.tiktok.com/vt.tiktok.com share link.")
        video_id = match.group(1)
        canonical = f"https://www.tiktok.com/@_/video/{video_id}"
    else:
        match = re.fullmatch(r"/(?:explore|discovery/item)/([a-fA-F0-9]+)", path)
        if not match:
            raise SourceIngestError("unsupported_url", "Provide a Xiaohongshu/RedNote video item or xhslink.com share link.")
        video_id = match.group(1).lower()
        # The official XiaoHongShu extractor accepts www.xiaohongshu.com only.
        canonical = f"https://www.xiaohongshu.com/discovery/item/{video_id}"
    return SourceURL(provider, canonical + (f"?{query}" if query else ""), canonical, video_id)


def source_project_name(value: str) -> str:
    """Stable, filesystem-safe name that includes the video identity."""
    return normalize_source_url(value).identity.replace(":", "_")


def sanitized_source_url(value: str) -> str:
    return normalize_source_url(value).canonical_url


def download_source(url: str, output_dir: str, download_height: str | int = "1080", cookies: str | None = None) -> str:
    """Download through the existing yt-dlp service without duplicating it."""
    from .youtube import download_video
    return download_video(url, output_dir, download_height, cookies)


def _ensure_public_host(host: str) -> None:
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        raise SourceIngestError("network_error", "Unable to resolve the video platform host.") from None
    if not addresses or any(not ipaddress.ip_address(item[4][0].split("%")[0]).is_global for item in addresses):
        raise SourceIngestError("unsafe_url", "Video links must resolve to public internet addresses.")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def resolve_source_url(value: str, *, timeout: float = 15, max_redirects: int = 5) -> SourceURL:
    """Follow ordinary HTTP share redirects, checking every destination first.

    No HTML/JavaScript execution, login, cookies, or protection workarounds are
    used. Landing pages that do not expose a supported video URL fail clearly.
    """
    source = normalize_source_url(value)
    _ensure_public_host(urlsplit(source.url).hostname)
    if not source.is_short_link:
        return source
    current = source.url
    opener = build_opener(_NoRedirect())
    seen = set()
    for _ in range(max_redirects + 1):
        parsed, host = _parse_supported_url(current)
        if _HOSTS[host] != source.provider or current in seen:
            raise SourceIngestError("unsafe_url", "The share link has an invalid or looping redirect.")
        seen.add(current)
        _ensure_public_host(host)
        request = Request(current, headers={"User-Agent": "DubbingStory/1.0", "Accept": "text/html"})
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                location = exc.headers.get("Location")
                exc.close()
                if not location:
                    raise SourceIngestError("unsupported_url", "The share redirect has no destination.") from None
                destination = urljoin(current, location)
                dest_parsed, dest_host = _parse_supported_url(destination)
                if _HOSTS[dest_host] != source.provider:
                    raise SourceIngestError("unsafe_url", "The share link redirects outside its video platform.")
                _ensure_public_host(dest_host)
                try:
                    resolved = normalize_source_url(destination)
                except SourceIngestError:
                    raise SourceIngestError("restricted", "The platform redirected to a login or unsupported landing page. Use an accessible video URL or local file.") from None
                if not resolved.is_short_link:
                    # Normalization may map a RedNote alias to Xiaohongshu.
                    # Validate the actual extractor destination as well.
                    _ensure_public_host(urlsplit(resolved.url).hostname)
                    return resolved
                current = destination
                continue
            exc.close()
            category = "restricted" if exc.code in {401, 403, 429} else "unavailable"
            raise SourceIngestError(category, "The platform did not allow this share link to resolve. Use an accessible video URL or local file.") from None
        except (URLError, TimeoutError, OSError):
            raise SourceIngestError("network_error", "Unable to resolve the share link; check connectivity or use the canonical video URL.") from None
        else:
            response.close()
            raise SourceIngestError("unsupported_url", "The share link requires a browser landing page. Supply its canonical video URL or a local video.")
    raise SourceIngestError("unsupported_url", "The share link exceeded the redirect limit.")


def classify_download_error(exc: Exception) -> SourceIngestError:
    """Never echo raw yt-dlp errors: they may contain signed URLs or tokens."""
    text = str(exc).lower()
    if any(word in text for word in ("drm", "protected content")):
        return SourceIngestError("protected", "DRM/protected media is not supported. Supply a permitted local video.")
    if any(word in text for word in ("requested format is not available", "only images are available", "no video formats")):
        return SourceIngestError("no_video", "No accessible video stream was found; image posts and unsupported formats are not supported.")
    if any(word in text for word in ("private", "deleted", "removed", "unavailable", "404")):
        return SourceIngestError("unavailable", "This video is private, removed, or unavailable.")
    if any(word in text for word in ("login", "sign in", "cookies", "authentication", "401", "403", "429", "captcha", "blocked", "restricted")):
        return SourceIngestError("restricted", "The platform requires access or is restricting requests. No authentication/protection bypass was attempted; use an accessible URL or permitted local video.")
    if any(word in text for word in ("timed out", "timeout", "connection", "resolve", "network")):
        return SourceIngestError("network_error", "The platform could not be reached. Check connectivity and try again later.")
    if "no video formats" in text:
        return SourceIngestError("no_video", "No accessible video stream was found; image posts/carousels and restricted media are unsupported.")
    if "unsupported url" in text:
        return SourceIngestError("unsupported_url", "The installed yt-dlp extractor does not support this URL. Use a canonical video URL or permitted local video.")
    return SourceIngestError("extractor_error", "The installed yt-dlp extractor could not read this video. Check its platform support or use a permitted local video.")
