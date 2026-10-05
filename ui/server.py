"""Local browser interface for the trained SkeAI tiny model."""

from __future__ import annotations

import argparse
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from src.skeai.language_model import TinyCharacterLanguageModel

UI_DIR = Path(__file__).resolve().parent
ROOT = UI_DIR.parent
DEFAULT_CHECKPOINT = ROOT / "models" / "tiny_character_model.json"


class SkeAIService:
    def __init__(self, checkpoint: Path) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}\n"
                "Train the tiny model first or pass --checkpoint."
            )
        self.checkpoint = checkpoint
        self.model = TinyCharacterLanguageModel.load_checkpoint(checkpoint)
        self.lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "checkpoint": str(self.checkpoint),
            "vocabulary_size": self.model.tokenizer.vocab_size,
            "context_length": self.model.context_length,
            "hidden_size": self.model.hidden_size,
            "parameter_count": self.model.network.parameter_count(),
        }

    def chat(self, message: str, *, max_new_tokens: int = 64, temperature: float = 0.85) -> str:
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if not message.strip():
            raise ValueError("message cannot be empty")
        if max_new_tokens <= 0 or max_new_tokens > 256:
            raise ValueError("max_new_tokens must be between 1 and 256")
        if temperature <= 0.0 or temperature > 2.0:
            raise ValueError("temperature must be between 0 and 2")

        with self.lock:
            generated = self.model.generate(
                message,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=None,
                top_k=8,
                repetition_penalty=1.12,
                no_repeat_ngram_size=3,
            )

        if generated.startswith(message):
            return generated[len(message):]
        return generated


class RequestHandler(BaseHTTPRequestHandler):
    service: SkeAIService | None = None

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 64 * 1024:
            raise ValueError("Request body is empty or too large")
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send_file("index.html", "text/html; charset=utf-8")
            return
        if self.path == "/style.css":
            self._send_file("style.css", "text/css; charset=utf-8")
            return
        if self.path == "/app.js":
            self._send_file("app.js", "application/javascript; charset=utf-8")
            return
        if self.path == "/api/status":
            assert self.service is not None
            self._send_json(self.service.status())
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            assert self.service is not None
            payload = self._read_json()
            response = self.service.chat(
                payload.get("message", ""),
                max_new_tokens=int(payload.get("max_new_tokens", 64)),
                temperature=float(payload.get("temperature", 0.85)),
            )
            self._send_json({"response": response})
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _send_file(self, filename: str, content_type: str) -> None:
        path = UI_DIR / filename
        if not path.is_file():
            self._send_json({"error": "UI file not found"}, HTTPStatus.NOT_FOUND)
            return
        self._send_bytes(path.read_bytes(), content_type)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[SkeAI UI] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local SkeAI web UI.")
    parser.add_argument("--host", default=os.environ.get("SKEAI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SKEAI_PORT", "8080")))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("SKEAI_CHECKPOINT", str(DEFAULT_CHECKPOINT))),
    )
    args = parser.parse_args()

    service = SkeAIService(args.checkpoint.resolve())
    RequestHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"SkeAI UI: http://{args.host}:{args.port}")
    print(f"Checkpoint: {service.checkpoint}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nStopping SkeAI UI...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
