"""
dubbingstory.tts.piper_tts_engine — Piper TTS engine implementation
"""

import os
import subprocess
import glob
from dubbingstory.tts.base_engine import BaseTTSEngine

class PiperTTSEngine(BaseTTSEngine):
    def __init__(self, custom_voices: dict | None = None, models_dir: str | None = None):
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "piper")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default Piper voices
        self.voices = {
            "id": "id_ID-news_tts-medium",
            "en": "en_US-lessac-medium", 
        }
        if custom_voices:
            self.voices.update(custom_voices)

    def get_engine_name(self) -> str:
        return "piper"

    def list_voices(self, language: str = "id") -> list[dict]:
        return [
            {"voice_id": self.voices.get("id"), "description": "Piper TTS Indonesian Voice"},
            {"voice_id": self.voices.get("en"), "description": "Piper TTS English Voice"},
        ]

    def _ensure_model_downloaded(self, voice_id: str) -> str:
        """Ensure the .onnx and .json model files are downloaded."""
        direct_path = os.path.join(self.models_dir, f"{voice_id}.onnx")
        subfolder_path = os.path.join(self.models_dir, voice_id, f"{voice_id}.onnx")
        
        if os.path.exists(direct_path):
            return direct_path
        if os.path.exists(subfolder_path):
            return subfolder_path
            
        print(f"      ⬇️ Downloading Piper model '{voice_id}' to {self.models_dir}...")
        try:
            subprocess.run(
                [
                    "python", "-m", "piper.download_voices",
                    voice_id,
                    "--data-dir", self.models_dir
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"      ⚠️ Failed to download Piper voice {voice_id}.")
            raise RuntimeError(f"Piper download failed for {voice_id}") from e
            
        if os.path.exists(direct_path):
            return direct_path
        if os.path.exists(subfolder_path):
            return subfolder_path
            
        # fallback search
        found = glob.glob(os.path.join(self.models_dir, "**", f"{voice_id}.onnx"), recursive=True)
        if found:
            return found[0]
            
        raise RuntimeError(f"Could not locate {voice_id}.onnx after download.")

    def synthesize(
        self,
        text: str,
        output_path: str,
        language: str = "id",
        voice_id: str | None = None,
        speaking_rate: float = 1.0,
    ) -> str:
        if not voice_id:
            voice_id = self.voices.get(language, self.voices["id"])
            
        model_path = self._ensure_model_downloaded(voice_id)
        
        cmd = [
            "python", "-m", "piper",
            "-m", model_path,
            "-f", output_path
        ]
        
        # apply length scale for speaking rate
        if speaking_rate != 1.0:
            length_scale = max(0.5, min(2.0, 1.0 / speaking_rate))
            cmd.extend(["--length_scale", f"{length_scale:.2f}"])
            
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            _, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                raise RuntimeError(f"Piper error: {stderr}")
                
        except Exception as e:
            raise RuntimeError(f"Piper synthesis failed: {e}")
            
        return output_path
