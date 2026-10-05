"""Local browser interface for a trained SkeAI Level 2 Transformer."""

from __future__ import annotations

import argparse
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from src.skeai.level2.chat import SkeAIConversation
from src.skeai.level2.tokenizer import HybridTokenizer
from src.skeai.level2.transformer import TinyTransformerLM


UI_DIR = Path(__file__).resolve().parent
ROOT = UI_DIR.parent
DEFAULT_CHECKPOINT = ROOT / "models" / "level2_transformer.json"
DEFAULT_TOKENIZER = ROOT / "models" / "level2_tokenizer.json"
DEFAULT_MEMORY = ROOT / "models" / "conversation_memory.json"


class SkeAILevel2Service:
    def __init__(
        self,
        checkpoint: Path,
        tokenizer_path: Path,
        memory_path: Path,
    ) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Level 2 checkpoint not found: {checkpoint}\n"
                "Train Level 2 first or pass --checkpoint."
            )
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Level 2 tokenizer not found: {tokenizer_path}\n"
                "Train Level 2 first or pass --tokenizer."
            )

        self.checkpoint = checkpoint
        self.tokenizer_path = tokenizer_path
        self.memory_path = memory_path
        self.model = TinyTransformerLM.load_checkpoint(checkpoint)
        self.tokenizer = HybridTokenizer.load(tokenizer_path)

        if self.model.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                "Checkpoint and tokenizer vocabulary sizes do not match."
            )

        self.conversation = SkeAIConversation(
            self.model,
            self.tokenizer,
            memory_path=memory_path,
            max_history_turns=3,
        )
        self.lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "model_type": "level2_transformer",
            "checkpoint": str(self.checkpoint),
            "tokenizer": str(self.tokenizer_path),
            "memory": str(self.memory_path),
            "vocabulary_size": self.tokenizer.vocab_size,
            "context_length": self.model.config.context_length,
            "d_model": self.model.config.d_model,
            "heads": self.model.config.n_heads,
            "layers": self.model.config.n_layers,
            "parameter_count": self.model.parameter_count(),
            "conversation_turns": len(self.conversation.memory.turns),
            "remembered_facts": self.conversation.memory.facts,
        }

    def chat(
        self,
        message: str,
        *,
        max_new_tokens: int = 48,
        temperature: float = 0.35,
        top_k: int = 8,
    ) -> str:
        if max_new_tokens <= 0 or max_new_tokens > 256:
            raise ValueError("max_new_tokens must be between 1 and 256")
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if top_k < 0 or top_k > self.tokenizer.vocab_size:
            raise ValueError("top_k must be between 0 and the vocabulary size")

        with self.lock:
            return self.conversation.chat(
                message,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=1234,
            )

    def reset_conversation(self, *, clear_memory: bool = False) -> None:
        with self.lock:
            self.conversation.reset(
                clear_persistent_memory=clear_memory,
            )


class RequestHandler(BaseHTTPRequestHandler):
    service: SkeAILevel2Service | None = None

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: int = 200,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(
        self,
        payload: dict[str, Any],
        status: int = 200,
    ) -> None:
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
        if self.path == "/api/chat":
            try:
                assert self.service is not None
                payload = self._read_json()
                response = self.service.chat(
                    payload.get("message", ""),
                    max_new_tokens=int(payload.get("max_new_tokens", 48)),
                    temperature=float(payload.get("temperature", 0.35)),
                    top_k=int(payload.get("top_k", 8)),
                )
                self._send_json({"response": response})
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover
                self._send_json(
                    {"error": str(exc)},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if self.path == "/api/reset":
            try:
                assert self.service is not None
                payload = self._read_json()
                self.service.reset_conversation(
                    clear_memory=bool(payload.get("clear_memory", False)),
                )
                self._send_json({"ok": True})
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _send_file(self, filename: str, content_type: str) -> None:
        path = UI_DIR / filename
        if not path.is_file():
            self._send_json(
                {"error": "UI file not found"},
                HTTPStatus.NOT_FOUND,
            )
            return
        self._send_bytes(path.read_bytes(), content_type)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[SkeAI Level 2 UI] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SkeAI Level 2 local web UI."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("SKEAI_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SKEAI_PORT", "8080")),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            os.environ.get(
                "SKEAI_LEVEL2_CHECKPOINT",
                str(DEFAULT_CHECKPOINT),
            )
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            os.environ.get(
                "SKEAI_LEVEL2_TOKENIZER",
                str(DEFAULT_TOKENIZER),
            )
        ),
    )
    parser.add_argument(
        "--memory",
        type=Path,
        default=Path(
            os.environ.get(
                "SKEAI_MEMORY",
                str(DEFAULT_MEMORY),
            )
        ),
    )
    args = parser.parse_args()

    service = SkeAILevel2Service(
        args.checkpoint.resolve(),
        args.tokenizer.resolve(),
        args.memory.resolve(),
    )
    RequestHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)

    print(f"SkeAI Level 2 UI: http://{args.host}:{args.port}")
    print(f"Checkpoint: {service.checkpoint}")
    print(f"Tokenizer: {service.tokenizer_path}")
    print(f"Memory: {service.memory_path}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SkeAI Level 2 UI...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
