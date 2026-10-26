"""Offline ingest regressions. No real platform, FFmpeg, or browser access."""

import hashlib
import io
import json
import os
import socket
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from urllib.error import HTTPError
from unittest.mock import patch

from dubbingstory.ingest import local_video, sources, youtube


REDNOTE_ID = "6a47c2cc000000001700a073"
REDNOTE_URL = f"https://www.rednote.com/discovery/item/{REDNOTE_ID}"
XHS_URL = f"https://www.xiaohongshu.com/discovery/item/{REDNOTE_ID}"
YOUTUBE_URL = "https://www.youtube.com/watch?v=abcdefghijk"
TIKTOK_URL = "https://www.tiktok.com/@creator/video/1234567890123456789"


class SourceURLTests(unittest.TestCase):
    def test_rednote_alias_preserves_extractor_query_only_in_memory(self):
        source = sources.normalize_source_url(REDNOTE_URL + "?xsec_token=example-only")
        self.assertEqual(source.provider, "xiaohongshu")
        self.assertEqual(source.video_id, REDNOTE_ID)
        self.assertEqual(source.canonical_url, XHS_URL)
        self.assertIn("xsec_token=example-only", source.url)
        self.assertNotIn("example-only", repr(source))
        self.assertNotIn("example-only", source.identity)
        self.assertEqual(sources.source_project_name(REDNOTE_URL), f"xiaohongshu_{REDNOTE_ID}")

    def test_youtube_variants_share_identity_but_different_videos_do_not(self):
        urls = [YOUTUBE_URL + "&list=discarded", "https://youtu.be/abcdefghijk?t=10",
                "https://m.youtube.com/shorts/abcdefghijk"]
        self.assertEqual({sources.normalize_source_url(url).identity for url in urls},
                         {"youtube:abcdefghijk"})
        self.assertNotEqual(sources.source_project_name(urls[0]),
                            sources.source_project_name("https://youtu.be/lmnopqrstuv"))

    def test_share_message_and_douyin_canonical(self):
        source = sources.normalize_source_url("Lihat video ini:\nhttps://v.douyin.com/abc123/。")
        self.assertTrue(source.is_short_link)
        self.assertEqual(source.canonical_url, "https://v.douyin.com/abc123")
        direct = sources.normalize_source_url("https://www.iesdouyin.com/share/video/123456789/")
        self.assertEqual(direct.canonical_url, "https://www.douyin.com/video/123456789")

    def test_tiktok_video_variants_use_shared_extractor(self):
        source = sources.normalize_source_url(TIKTOK_URL)
        self.assertEqual(source.provider, "tiktok")
        self.assertEqual(source.extractor_key, "TikTok")
        self.assertEqual(source.video_id, "1234567890123456789")
        self.assertEqual(
            source.canonical_url,
            "https://www.tiktok.com/@_/video/1234567890123456789",
        )
        short = sources.normalize_source_url("https://vm.tiktok.com/ZMexample/")
        self.assertTrue(short.is_short_link)
        self.assertEqual(short.provider, "tiktok")

    def test_untrusted_hosts_credentials_and_nonvideo_urls_rejected(self):
        for url in ["file:///video.mp4", "https://127.0.0.1/video/123",
                    "https://www.rednote.com.evil.example/discovery/item/" + REDNOTE_ID,
                    "https://name:pass@www.rednote.com/discovery/item/" + REDNOTE_ID,
                    "https://www.rednote.com:8080/discovery/item/" + REDNOTE_ID,
                    "https://www.youtube.com/playlist?list=abc",
                    "https://www.xiaohongshu.com/user/profile/123",
                    REDNOTE_URL + " " + YOUTUBE_URL]:
            with self.subTest(url=url), self.assertRaises(sources.SourceIngestError):
                sources.normalize_source_url(url)

    @patch.object(sources.socket, "getaddrinfo")
    def test_private_dns_answers_rejected(self, dns):
        dns.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        with self.assertRaises(sources.SourceIngestError) as error:
            sources.resolve_source_url(REDNOTE_URL)
        self.assertEqual(error.exception.category, "unsafe_url")

    @patch.object(sources, "_ensure_public_host")
    @patch.object(sources, "build_opener")
    def test_share_redirect_normalizes_rednote_and_checks_actual_destination(self, build, public):
        build.return_value.open.side_effect = HTTPError(
            "https://xhslink.com/a/123", 302, "Found", {"Location": REDNOTE_URL}, None)
        resolved = sources.resolve_source_url("https://xhslink.com/a/123")
        self.assertEqual(resolved.canonical_url, XHS_URL)
        public.assert_any_call("www.xiaohongshu.com")
        self.assertEqual(build.return_value.open.call_count, 1)

    @patch.object(sources, "_ensure_public_host")
    @patch.object(sources, "build_opener")
    def test_redirect_cannot_escape_platform_or_reach_loopback(self, build, public):
        for target in [YOUTUBE_URL, "http://127.0.0.1/private"]:
            build.return_value.open.side_effect = HTTPError(
                "https://xhslink.com/a/123", 302, "Found", {"Location": target}, None)
            with self.subTest(target=target), self.assertRaises(sources.SourceIngestError):
                sources.resolve_source_url("https://xhslink.com/a/123")

    @patch.object(sources, "_ensure_public_host")
    @patch.object(sources, "build_opener")
    def test_redirect_loop_and_login_fail_without_browser(self, build, public):
        for target in ["https://xhslink.com/a/123", "https://www.xiaohongshu.com/login"]:
            build.return_value.open.side_effect = HTTPError(
                "https://xhslink.com/a/123", 302, "Found", {"Location": target}, None)
            with self.subTest(target=target), self.assertRaises(sources.SourceIngestError):
                sources.resolve_source_url("https://xhslink.com/a/123")

    def test_downloader_error_messages_never_echo_raw_url(self):
        error = sources.classify_download_error(RuntimeError("HTTP 403 https://cdn.example/video?signature=example-only"))
        self.assertEqual(error.category, "restricted")
        self.assertNotIn("example-only", str(error))
        self.assertNotIn("cdn.example", str(error))

    def test_format_failure_is_not_misreported_as_missing_or_private_video(self):
        error = sources.classify_download_error(RuntimeError("Requested format is not available"))
        self.assertEqual(error.category, "format_unavailable")
        self.assertNotIn("private", str(error))
        self.assertNotIn("image posts", str(error))

    def test_actual_missing_formats_remain_distinct_from_selection_failure(self):
        for message in ("Only images are available for download", "No video formats found"):
            with self.subTest(message=message):
                error = sources.classify_download_error(RuntimeError(message))
                self.assertEqual(error.category, "no_video")
                self.assertNotIn("image posts", str(error))

    def test_explicit_access_requirement_takes_priority_over_generic_unavailable(self):
        error = sources.classify_download_error(
            RuntimeError("Video unavailable. Sign in to confirm you're not a bot. HTTP 403"))
        self.assertEqual(error.category, "restricted")


class DownloaderDiagnosticsTests(unittest.TestCase):
    def test_challenge_warning_does_not_claim_ejs_is_missing_from_generic_advice(self):
        logger = youtube._SafeLogger()
        logger.warning(
            "n challenge solving failed: Some formats may be missing. "
            "Ensure you have a supported JavaScript runtime and challenge solver "
            "script distribution installed. See https://github.com/yt-dlp/yt-dlp/wiki/EJS")
        self.assertEqual(logger.codes, {"js_challenge_failed"})

    def test_diagnostic_keywords_inside_a_signed_url_do_not_become_reason_codes(self):
        logger = youtube._SafeLogger()
        logger.error("Extractor failed https://cdn.example/ejs?reason=unavailable&po_token=fixture")
        self.assertEqual(logger.codes, {"extractor_error"})

    def test_recognized_warnings_keep_only_safe_reason_codes(self):
        cases = [
            ("No supported JavaScript runtime could be found", "js_runtime_unavailable"),
            ("n challenge solving failed", "js_challenge_failed"),
            ("Signature solving failed", "js_challenge_failed"),
            ("EJS challenge solver scripts are unavailable", "ejs_unavailable"),
            ("This client requires a PO Token", "platform_access_required"),
            ("Sign in to confirm you're not a bot", "platform_access_required"),
            ("HTTP Error 429: Too Many Requests", "platform_access_required"),
            ("Some formats are missing a url. YouTube is forcing SABR streaming", "youtube_delivery_unavailable"),
            ("Only images are available for download", "no_video_formats"),
            ("Requested format is not available", "format_selection"),
        ]
        for message, code in cases:
            with self.subTest(code=code, message=message):
                logger = youtube._SafeLogger()
                raw = message + " https://cdn.example/video?token=private-fixture-value"
                captured = io.StringIO()
                with redirect_stdout(captured), redirect_stderr(captured):
                    logger.warning(raw)
                    logger.warning(raw)
                    logger.report()
                self.assertIn(code, logger.codes)
                self.assertTrue(captured.getvalue().strip())
                for unsafe in ("cdn.example", "private-fixture-value"):
                    self.assertNotIn(unsafe, captured.getvalue())
                    self.assertNotIn(unsafe, repr(vars(logger)))

    def test_unknown_messages_and_debug_output_never_leak_or_accumulate(self):
        logger = youtube._SafeLogger()
        raw = "opaque-fixture-diagnostic https://cdn.example/file?signature=private-fixture-value"
        captured = io.StringIO()
        with redirect_stdout(captured), redirect_stderr(captured):
            logger.debug(raw)
            logger.info(raw)
            self.assertEqual(logger.codes, set())
            logger.warning(raw)
            logger.error(raw)
            self.assertEqual(captured.getvalue(), "")
            logger.report()
        self.assertEqual(logger.codes, {"extractor_warning", "extractor_error"})
        self.assertTrue(captured.getvalue().strip())
        for unsafe in ("opaque-fixture-diagnostic", "cdn.example", "private-fixture-value"):
            self.assertNotIn(unsafe, captured.getvalue())
            self.assertNotIn(unsafe, repr(vars(logger)))


class DownloadBackendTests(unittest.TestCase):
    def test_new_sources_do_not_read_cookies_or_use_youtube_options(self):
        with patch.object(youtube, "_apply_cookies") as cookies:
            opts = youtube._base_options("xiaohongshu", "unused-browser")
        cookies.assert_not_called()
        self.assertNotIn("extractor_args", opts)
        self.assertNotIn("cookiesfrombrowser", opts)
        self.assertFalse(opts["allow_unplayable_formats"])
        self.assertFalse(opts["cachedir"])

    def test_youtube_uses_upstream_clients_and_keeps_existing_cookie_support(self):
        with patch.object(youtube, "_apply_cookies") as cookies:
            opts = youtube._base_options("youtube", "fixture-browser")
        cookies.assert_called_once_with(opts, "fixture-browser")
        self.assertNotIn("extractor_args", opts)
        self.assertEqual(opts["remote_components"], ["ejs:github"])
        self.assertFalse(opts["no_warnings"])
        self.assertIsInstance(opts["logger"], youtube._SafeLogger)

    @patch.object(youtube, "YoutubeDL")
    def test_subtitle_discovery_and_transfer_do_not_reintroduce_client_pins(self, downloader):
        session = downloader.return_value.__enter__.return_value
        session.extract_info.return_value = {"subtitles": {"id": [{"ext": "srt"}]}}
        with tempfile.TemporaryDirectory() as directory:
            expected = os.path.join(directory, "yt_subs.id.srt")

            def transfer(urls):
                with open(expected, "w", encoding="utf-8") as handle:
                    handle.write("1\n00:00:00,000 --> 00:00:01,000\nFixture\n")

            session.download.side_effect = transfer
            self.assertEqual(youtube.download_subtitles(YOUTUBE_URL, directory, ["id"]), expected)
        self.assertEqual(downloader.call_count, 2)
        for call in downloader.call_args_list:
            opts = call.args[0]
            self.assertNotIn("extractor_args", opts)
            self.assertFalse(opts["no_warnings"])
            self.assertIsInstance(opts["logger"], youtube._SafeLogger)

    @patch.object(youtube, "resolve_source_url")
    @patch.object(youtube, "YoutubeDL")
    def test_format_failure_reports_safe_context_without_publishing_source(self, downloader, resolve):
        resolve.return_value = sources.normalize_source_url(YOUTUBE_URL)
        session = downloader.return_value.__enter__.return_value
        raw_url = "https://cdn.example/video?token=private-fixture-value"

        def fail(*args, **kwargs):
            logger = downloader.call_args.args[0]["logger"]
            logger.warning("Signature solving failed " + raw_url)
            logger.error("Requested format is not available " + raw_url)
            raise RuntimeError("Requested format is not available " + raw_url)

        session.extract_info.side_effect = fail
        captured = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(youtube._SafeLogger, "report", autospec=True) as report:
                with redirect_stdout(captured), redirect_stderr(captured):
                    with self.assertRaises(sources.SourceIngestError) as error:
                        youtube.download_video(YOUTUBE_URL, directory)
                report.assert_called_once()
            self.assertEqual(error.exception.category, "format_unavailable")
            self.assertEqual(os.listdir(directory), [])
        session.process_ie_result.assert_not_called()
        logger = downloader.call_args.args[0]["logger"]
        self.assertIn("js_challenge_failed", logger.codes)
        self.assertIn("format_selection", logger.codes)
        for unsafe in ("cdn.example", "private-fixture-value"):
            self.assertNotIn(unsafe, captured.getvalue())
            self.assertNotIn(unsafe, str(error.exception))
            self.assertNotIn(unsafe, repr(vars(logger)))

    def test_quality_keeps_legacy_height_limit_and_supports_silent_video(self):
        limited = youtube._build_format_selector("1080")
        self.assertIn("height<=?1080", limited)
        self.assertTrue(limited.split("/")[-1].startswith("bestvideo"))
        native = youtube._build_format_selector("max")
        self.assertNotIn("height", native)
        self.assertTrue(native.split("/")[-1].startswith("bestvideo"))

    def test_playlists_live_drm_and_audio_only_rejected(self):
        cases = [
            ({"_type": "playlist", "entries": []}, "unsupported_media"),
            ({"formats": [{"url": "https://media.example/file", "vcodec": "h264"}], "is_live": True}, "unsupported_media"),
            ({"formats": [{"url": "https://media.example/file", "has_drm": True}]}, "protected"),
            ({"formats": [{"url": "https://media.example/file", "vcodec": "none"}]}, "no_video"),
        ]
        for info, category in cases:
            with self.subTest(category=category), self.assertRaises(sources.SourceIngestError) as error:
                youtube._require_single_video(info)
            self.assertEqual(error.exception.category, category)

    def test_metadata_has_safe_identity_and_rejects_wrong_video(self):
        source = sources.normalize_source_url(REDNOTE_URL)
        info = {"id": REDNOTE_ID, "description": "See https://example.org/view?token=example-only",
                "width": 1080, "height": 1920, "vcodec": "h264", "url": "https://cdn.example/signed"}
        metadata = youtube._metadata(info, source)
        encoded = json.dumps(metadata)
        self.assertNotIn("example-only", encoded)
        self.assertNotIn("cdn.example", encoded)
        self.assertEqual(metadata["source_identity"], f"xiaohongshu:{REDNOTE_ID}")
        with self.assertRaises(sources.SourceIngestError):
            youtube._metadata({**info, "id": "different"}, source)

    def test_cache_requires_matching_identity_quality_and_content(self):
        source = sources.normalize_source_url(REDNOTE_URL)
        with tempfile.TemporaryDirectory() as directory:
            media = os.path.join(directory, "source.mp4")
            with open(media, "wb") as handle:
                handle.write(b"fixture-media")
            metadata = {"request_identity": source.identity, "download_quality": "max",
                        "sha256": hashlib.sha256(b"fixture-media").hexdigest()}
            with open(os.path.join(directory, "video_metadata.json"), "w") as handle:
                json.dump(metadata, handle)
            self.assertEqual(youtube._reusable_source(directory, source, "max"), media)
            for requested, quality in [(source, "1080"), (sources.normalize_source_url(YOUTUBE_URL), "max")]:
                with self.assertRaises(sources.SourceIngestError) as error:
                    youtube._reusable_source(directory, requested, quality)
                self.assertEqual(error.exception.category, "cache_conflict")
            with open(media, "ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(sources.SourceIngestError) as error:
                youtube._reusable_source(directory, source, "max")
            self.assertEqual(error.exception.category, "cache_invalid")

    def test_missing_cached_video_is_not_replaced_under_stale_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "video_metadata.json"), "w") as handle:
                json.dump({"provider": "xiaohongshu"}, handle)
            with self.assertRaises(sources.SourceIngestError) as error:
                youtube._reusable_source(directory, sources.normalize_source_url(REDNOTE_URL), "max")
            self.assertEqual(error.exception.category, "cache_invalid")

    @patch.object(youtube, "resolve_source_url")
    def test_new_share_link_can_reuse_same_resolved_video(self, resolve):
        resolved = sources.normalize_source_url(REDNOTE_URL)
        resolve.return_value = resolved
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "source.mp4")
            with open(output, "wb") as handle:
                handle.write(b"fixture-video")
            metadata = {"source_identity": resolved.identity, "download_quality": "max",
                        "sha256": hashlib.sha256(b"fixture-video").hexdigest()}
            with open(os.path.join(directory, "video_metadata.json"), "w") as handle:
                json.dump(metadata, handle)
            self.assertEqual(youtube.download_video("https://xhslink.com/a/new-share", directory, "max"), output)
            resolve.assert_called_once()

    @patch.object(youtube, "resolve_source_url")
    @patch.object(youtube, "YoutubeDL")
    def test_failed_transfer_never_publishes_partial_source(self, downloader, resolve):
        resolve.return_value = sources.normalize_source_url(REDNOTE_URL)
        session = downloader.return_value.__enter__.return_value
        session.extract_info.return_value = {
            "id": REDNOTE_ID, "formats": [{"url": "https://media.example/file", "vcodec": "h264"}]}
        session.process_ie_result.side_effect = RuntimeError("HTTP 403")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(sources.SourceIngestError):
                youtube.download_video(REDNOTE_URL, directory, "max")
            self.assertEqual(os.listdir(directory), [])

    @patch.object(local_video, "get_video_metadata")
    @patch.object(youtube, "resolve_source_url")
    @patch.object(youtube, "YoutubeDL")
    def test_success_publishes_actual_geometry_hash_and_reuses_without_network(self, downloader, resolve, probe):
        resolve.return_value = sources.normalize_source_url(REDNOTE_URL)
        info = {"id": REDNOTE_ID, "width": 720, "height": 1280,
                "formats": [{"url": "https://media.example/file", "vcodec": "h264"}]}
        session = downloader.return_value.__enter__.return_value
        session.extract_info.return_value = info
        probe.return_value = {"width": 1080, "height": 1920, "video_duration": 10, "has_audio": False}

        def transfer(extracted, download):
            template = downloader.call_args.args[0]["outtmpl"]
            with open(template.replace("%(ext)s", "mp4"), "wb") as handle:
                handle.write(b"fixture-video")
            return extracted

        session.process_ie_result.side_effect = transfer
        with tempfile.TemporaryDirectory() as directory:
            output = youtube.download_video(REDNOTE_URL, directory, "max")
            with open(os.path.join(directory, "video_metadata.json"), encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["height"], 1920)
            self.assertEqual(metadata["sha256"], hashlib.sha256(b"fixture-video").hexdigest())
            resolve.reset_mock()
            self.assertEqual(youtube.download_video(REDNOTE_URL, directory, "max"), output)
            resolve.assert_not_called()


class MediaMetadataTests(unittest.TestCase):
    def test_rotation_sar_stream_duration_and_audio_offset_are_retained(self):
        probe = {"format": {"duration": "12.5", "start_time": "0.4"}, "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "sample_aspect_ratio": "1:1",
             "side_data_list": [{"rotation": -90}], "duration_ts": 12000, "time_base": "1/1000",
             "start_time": "0.4", "r_frame_rate": "30/1", "avg_frame_rate": "29/1"},
            {"codec_type": "audio", "start_time": "0.6", "duration": "11.7"}]}
        metadata = local_video.metadata_from_probe(probe)
        self.assertEqual((metadata["display_width"], metadata["display_height"]), (1080, 1920))
        self.assertEqual(metadata["video_duration"], 12)
        self.assertEqual(metadata["duration"], 12.5)
        self.assertEqual(metadata["rotation"], 270)
        self.assertEqual(metadata["audio_start_time"], 0.6)
        self.assertTrue(metadata["vfr_possible"])

    def test_attached_art_is_not_selected_as_video_and_bad_rates_are_safe(self):
        metadata = local_video.metadata_from_probe({"format": {"duration": "3"}, "streams": [
            {"codec_type": "video", "width": 400, "height": 400, "disposition": {"attached_pic": 1}},
            {"codec_type": "video", "width": 720, "height": 1280, "avg_frame_rate": "0/0",
             "r_frame_rate": "N/A", "sample_aspect_ratio": "N/A"}]})
        self.assertEqual(metadata["width"], 720)
        self.assertEqual(metadata["video_duration_source"], "container")
        self.assertEqual(metadata["display_width"], 720)
        self.assertEqual(metadata["fps"], 0)
        self.assertFalse(metadata["has_audio"])

    @patch.object(local_video, "get_video_metadata")
    def test_local_source_conflict_never_overwrites_project(self, probe):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "input.mp4")
            project = os.path.join(directory, "project")
            os.mkdir(project)
            destination = os.path.join(project, "source.mp4")
            with open(source, "wb") as handle:
                handle.write(b"new-video")
            with open(destination, "wb") as handle:
                handle.write(b"existing-video")
            with self.assertRaises(sources.SourceIngestError) as error:
                local_video.validate_and_copy(source, project)
            self.assertEqual(error.exception.category, "cache_conflict")
            with open(destination, "rb") as handle:
                self.assertEqual(handle.read(), b"existing-video")
            probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
