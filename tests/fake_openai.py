"""OpenAI-compatible HTTP server replaying scripted responses (tests the real ChatOpenAI client)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def completion(content: str = "", tool_calls: list[tuple[str, dict]] | None = None) -> tuple[int, dict]:
    """A successful chat completion, optionally with tool calls given as (name, args)."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(tool_calls)
        ]
    finish = "tool_calls" if tool_calls else "stop"
    return 200, {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "fake",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def error(code: str, message: str = "bad request", **extra) -> tuple[int, dict]:
    """A 400 error in the shape Groq uses (e.g. code='tool_use_failed', failed_generation=...)."""
    return 400, {"error": {"message": message, "type": "invalid_request_error", "code": code, **extra}}


class FakeOpenAI:
    """Serves `responses` in order; every request body is kept in `requests`."""

    def __init__(self, responses: list[tuple[int, dict]]):
        self.responses = list(responses)
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                status, payload = fake.responses.pop(0) if fake.responses else error("exhausted", "no script")
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
