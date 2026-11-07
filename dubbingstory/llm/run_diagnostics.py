"""Run-scoped notebook diagnostics, available even when generation fails."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import zipfile

from dubbingstory.llm.diagnostics import LLMDiagnostics


_CREDENTIAL_KEY = re.compile(
    r"api.?key|password|passwd|authorization|cookie|secret|credential|^token$|^hf_token$|access.?token|refresh.?token", re.I)
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}\Z")


class RunDiagnostics:
    """Keep one run's settings, stages and traces together; never scan a project.

    Only explicitly registered logs are bundled. Raw notebook outputs, prompts,
    model access configuration, media and unrelated prior runs are not copied.
    """

    def __init__(self, project_dir, run_id: str, project: str, *, secrets=()):
        if not _SAFE_NAME.fullmatch(run_id) or run_id in {".", ".."}:
            raise ValueError("Invalid diagnostics run ID")
        self.project_dir = Path(project_dir).resolve()
        self.run_id = run_id
        self.directory = self.project_dir / "runs" / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.redactor = LLMDiagnostics(secrets=secrets)
        self.manifest = {"schema_version": 1, "run_id": run_id, "project": project,
                         "started_at": self._now(), "status": "running", "stages": [], "logs": []}
        self.trace_path = self.directory / "llm_attempts.jsonl"
        self._save()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def _safe_config(self, value):
        if isinstance(value, dict):
            return {str(key): ("[REDACTED]" if _CREDENTIAL_KEY.search(str(key))
                              else self._safe_config(item)) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._safe_config(item) for item in value]
        if isinstance(value, str):
            return self.redactor.redact(value)
        if value is None or type(value) in {bool, int, float}:
            return value
        return "[UNSUPPORTED VALUE]"

    def snapshot_config(self, config):
        (self.directory / "notebook_runtime.yaml").write_text(
            json.dumps(self._safe_config(config), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")

    def _save(self):
        (self.directory / "run_manifest.json").write_text(
            json.dumps(self._safe_config(self.manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def log_path(self, label: str) -> Path:
        if not _SAFE_NAME.fullmatch(label) or label in {".", ".."}:
            raise ValueError("Invalid diagnostics log label")
        name = label if label.endswith(".log") else f"notebook_{label}.log"
        if name not in self.manifest["logs"]:
            self.manifest["logs"].append(name)
            self._save()
        return self.directory / name

    def stage(self, stage: str, status: str):
        if status not in {"running", "complete", "failed"}:
            raise ValueError("Unknown diagnostics stage status")
        self.manifest["stages"].append({"stage": stage, "status": status, "timestamp": self._now()})
        self._save()

    def finish(self, *, success: bool, error_type: str = "") -> Path:
        """Make a small sanitized ZIP; callers keep the original pipeline error."""
        self.manifest.update(status="complete" if success else "failed", finished_at=self._now(),
                             error_type=self.redactor.safe_text(error_type, 128))
        self._save()
        target = self.project_dir / f"diagnostics_{self.run_id}.zip"
        names = ["run_manifest.json", "notebook_runtime.yaml", "llm_attempts.jsonl", *self.manifest["logs"]]
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in dict.fromkeys(names):
                path = self.directory / name
                if not path.is_file():
                    continue
                # Limit log retention, keeping the tail where failures appear.
                with path.open("rb") as handle:
                    length = handle.seek(0, 2)
                    handle.seek(max(0, length - 8 * 1024 * 1024))
                    data = handle.read().decode("utf-8", errors="replace")
                if length > 8 * 1024 * 1024:
                    data = data.partition("\n")[2]
                    self.manifest.setdefault("truncated_files", []).append(name)
                if name == "llm_attempts.jsonl":
                    rows = []
                    for line in data.splitlines():
                        try:
                            record = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(record, dict) and record.get("run_id") == self.run_id:
                            rows.append(self.redactor.redact(json.dumps(record, ensure_ascii=False)))
                    data = "\n".join(rows) + ("\n" if rows else "")
                else:
                    data = self.redactor.redact(data)
                if name != "run_manifest.json":
                    archive.writestr(name, data)
            self._save()
            archive.writestr("run_manifest.json", json.dumps(self._safe_config(self.manifest), ensure_ascii=False, indent=2))
        return target
