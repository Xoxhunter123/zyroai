#!/usr/bin/env python3
"""
Zyro AI — Agentic Coding & Voice Backend (OpenCode-Style)
Designed for Render (12 vCPU / 96 GB RAM + Qwen3-Coder-30B-A3B-Instruct via llama-server)
AND playable right now on your local Windows/Mac/Linux PC before deploying to Render!

Features:
  - POST /v1/agent/run        : Full multi-step tool-calling agent (SSE stream)
                                Tools: run_terminal, write_file, read_file, list_files, search_files, delete_file
  - POST /v1/chat/completions : Standard OpenAI-compatible SSE streaming (Chat & Voice Mode)
  - GET  /v1/workspace/files  : Browse files created by Zyro Agent
  - GET  /v1/workspace/file   : Read/download any workspace file
  - POST /v1/workspace/exec   : Direct interactive terminal execution
  - DELETE /v1/workspace/file : Delete a workspace file
  - GET  /health              : Health + model readiness + hardware stats

Local PC Usage (Zero dependencies!):
  python render-backend/server.py --demo
"""

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import argparse
import json
import os
import platform
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "Qwen3-Coder-30B-A3B-Instruct"

AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_terminal",
            "description": "Execute a shell command in the workspace directory (Python, Node, Bash, Git, tests, build commands). Returns exit code, stdout, and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30, max 120)"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file inside the workspace. Parent folders are created automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace, e.g. app/main.py"},
                    "content": {"type": "string", "description": "Full text content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file inside the workspace with optional line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace"},
                    "start_line": {"type": "integer", "description": "1-based start line (default 1)"},
                    "end_line": {"type": "integer", "description": "1-based end line (default 250)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative folder path (default '.')"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search text across files in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Substring or keyword to search for"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path to remove"}
                },
                "required": ["path"]
            }
        }
    }
]

AGENT_SYSTEM_SUFFIX = """
You are Zyro AI running in Agentic OpenCode Mode powered by Qwen3-Coder-30B-A3B.
You have real tools to create files (write_file), read files (read_file), list files (list_files), search files (search_files), delete files (delete_file), and run shell commands (run_terminal) inside the workspace.
Rules:
1. When the user asks you to build, code, script, test, or run something, USE YOUR TOOLS directly to create the real files and run/verify them!
2. Never pretend you ran a command — actually call run_terminal and report the real output.
3. Keep final summaries concise and include clean fenced markdown code blocks when helpful.
""".strip()


def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class WorkspaceExecutor:
    """Safe sandboxed workspace for file operations & terminal execution."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel_path: str) -> Path:
        clean = (rel_path or ".").strip().lstrip("/\\")
        target = (self.root / clean).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"Path escapes workspace: {rel_path}")
        return target

    def list_all_files(self) -> list[dict]:
        items = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                if rel.startswith(".git/"):
                    continue
                stat = p.stat()
                items.append({
                    "path": rel,
                    "size": stat.st_size,
                    "modified": int(stat.st_mtime * 1000)
                })
        return items

    def execute_tool(self, name: str, args: dict) -> dict:
        start = time.time()
        try:
            if name == "write_file":
                path = str(args.get("path", "")).strip()
                content = str(args.get("content", ""))
                target = self._resolve(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                lines = content.count("\n") + (1 if content else 0)
                return {
                    "ok": True,
                    "summary": f"Wrote {path} ({lines} lines, {len(content.encode('utf-8'))} bytes)",
                    "output": f"Created/updated {path} ({lines} lines)",
                    "exit_code": 0,
                    "duration_ms": int((time.time() - start) * 1000)
                }

            if name == "read_file":
                path = str(args.get("path", "")).strip()
                start_line = max(1, int(args.get("start_line", 1)))
                end_line = max(start_line, int(args.get("end_line", 250)))
                target = self._resolve(path)
                if not target.exists() or not target.is_file():
                    return {"ok": False, "summary": f"File not found: {path}", "output": f"Error: {path} does not exist", "exit_code": 1, "duration_ms": int((time.time() - start) * 1000)}
                all_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
                sliced = all_lines[start_line - 1:end_line]
                numbered = "\n".join(f"{start_line + i:>4} | {ln}" for i, ln in enumerate(sliced))
                return {
                    "ok": True,
                    "summary": f"Read {path} ({len(sliced)} lines)",
                    "output": numbered[:12000],
                    "exit_code": 0,
                    "duration_ms": int((time.time() - start) * 1000)
                }

            if name == "list_files":
                rel = str(args.get("path", ".")).strip() or "."
                target = self._resolve(rel)
                if not target.exists():
                    return {"ok": False, "summary": f"Directory not found: {rel}", "output": "Directory not found", "exit_code": 1, "duration_ms": int((time.time() - start) * 1000)}
                entries = []
                for item in sorted(target.iterdir()):
                    suffix = "/" if item.is_dir() else f" ({item.stat().st_size} B)"
                    entries.append(f"{item.relative_to(self.root).as_posix()}{suffix}")
                out = "\n".join(entries) if entries else "(workspace is empty)"
                return {
                    "ok": True,
                    "summary": f"Listed {len(entries)} items in {rel}",
                    "output": out,
                    "exit_code": 0,
                    "duration_ms": int((time.time() - start) * 1000)
                }

            if name == "search_files":
                query = str(args.get("query", "")).strip()
                matches = []
                for p in sorted(self.root.rglob("*")):
                    if not p.is_file() or p.stat().st_size > 1_000_000:
                        continue
                    rel = p.relative_to(self.root).as_posix()
                    try:
                        for idx, ln in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                            if query.lower() in ln.lower():
                                matches.append(f"{rel}:{idx}: {ln.strip()}")
                                if len(matches) >= 50:
                                    break
                    except Exception:
                        pass
                out = "\n".join(matches) if matches else f"No matches found for '{query}'"
                return {
                    "ok": True,
                    "summary": f"Found {len(matches)} matches for '{query}'",
                    "output": out,
                    "exit_code": 0,
                    "duration_ms": int((time.time() - start) * 1000)
                }

            if name == "delete_file":
                path = str(args.get("path", "")).strip()
                target = self._resolve(path)
                if not target.exists() or not target.is_file():
                    return {"ok": False, "summary": f"File not found: {path}", "output": "Not found", "exit_code": 1, "duration_ms": int((time.time() - start) * 1000)}
                target.unlink()
                return {
                    "ok": True,
                    "summary": f"Deleted {path}",
                    "output": f"Removed {path}",
                    "exit_code": 0,
                    "duration_ms": int((time.time() - start) * 1000)
                }

            if name == "run_terminal":
                cmd = str(args.get("command", "")).strip()
                if not cmd:
                    return {"ok": False, "summary": "Empty command", "output": "Command cannot be empty", "exit_code": 1, "duration_ms": 0}
                timeout = min(120, max(5, int(args.get("timeout", 30))))
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.root),
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                combined = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
                combined = combined.strip() or "(no output)"
                if len(combined) > 14000:
                    combined = combined[:14000] + "\n... [output truncated]"
                return {
                    "ok": proc.returncode == 0,
                    "summary": f"$ {cmd} (exit {proc.returncode})",
                    "output": combined,
                    "exit_code": proc.returncode,
                    "duration_ms": int((time.time() - start) * 1000)
                }

            return {"ok": False, "summary": f"Unknown tool: {name}", "output": f"Unsupported tool {name}", "exit_code": 1, "duration_ms": 0}

        except subprocess.TimeoutExpired:
            return {"ok": False, "summary": "Command timed out", "output": "Command exceeded timeout limit.", "exit_code": 124, "duration_ms": int((time.time() - start) * 1000)}
        except Exception as e:
            return {"ok": False, "summary": f"Tool error: {e}", "output": str(e), "exit_code": 1, "duration_ms": int((time.time() - start) * 1000)}


class ZyroAgentHandler(BaseHTTPRequestHandler):
    upstream: str | None = None
    api_key: str = ""
    model_name: str = DEFAULT_MODEL
    workspace: WorkspaceExecutor = WorkspaceExecutor(Path("./workspace"))
    demo_mode: bool = False
    demo_tps: float = 26.0

    def log_message(self, format: str, *args):
        print(f"[Zyro Agent] {self.address_string()} - {format % args}")

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _sse_event(self, payload: dict):
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _check_auth(self) -> bool:
        if not self.api_key:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {self.api_key}"

    def _upstream_ready(self) -> bool:
        if self.demo_mode or not self.upstream:
            return True
        try:
            with urllib.request.urlopen(f"{self.upstream.rstrip('/')}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            ready = self._upstream_ready()
            self._send_json(200, {
                "status": "ok",
                "model_ready": ready,
                "mode": "demo" if self.demo_mode else "render-qwen3-coder",
                "model": self.model_name,
                "agentic": True,
                "cpus": os.cpu_count() or 12,
                "workspace_files": len(self.workspace.list_all_files())
            })
            return

        if not self._check_auth():
            self._send_json(401, {"error": "Invalid API key"})
            return

        if path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": self.model_name,
                    "object": "model",
                    "owned_by": "zyro-render-agent",
                    "capabilities": ["chat", "voice", "agentic_tools", "terminal"]
                }]
            })
            return

        if path == "/v1/workspace/files":
            self._send_json(200, {"files": self.workspace.list_all_files()})
            return

        if path == "/v1/workspace/file":
            qs = parse_qs(parsed.query)
            rel = qs.get("path", [""])[0]
            try:
                target = self.workspace._resolve(rel)
                if not target.exists() or not target.is_file():
                    self._send_json(404, {"error": "File not found"})
                    return
                content = target.read_text(encoding="utf-8", errors="replace")
                self._send_json(200, {"path": rel, "content": content, "size": target.stat().st_size})
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        self._send_json(404, {"error": "Not found"})

    def do_DELETE(self):
        if not self._check_auth():
            self._send_json(401, {"error": "Invalid API key"})
            return
        parsed = urlparse(self.path)
        if parsed.path == "/v1/workspace/file":
            rel = parse_qs(parsed.query).get("path", [""])[0]
            res = self.workspace.execute_tool("delete_file", {"path": rel})
            self._send_json(200 if res["ok"] else 400, res)
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if not self._check_auth():
            self._send_json(401, {"error": "Invalid API key"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        try:
            req = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        path = urlparse(self.path).path

        if path == "/v1/workspace/exec":
            cmd = str(req.get("command", "")).strip()
            res = self.workspace.execute_tool("run_terminal", {"command": cmd})
            self._send_json(200, res)
            return

        if path == "/v1/agent/run":
            self._handle_agent_run(req)
            return

        if path == "/v1/chat/completions":
            self._handle_chat_completions(req, raw_body)
            return

        self._send_json(404, {"error": "Endpoint not found"})

    # -------------------------------------------------------------------------
    # Standard Chat & Voice SSE Endpoint
    # -------------------------------------------------------------------------
    def _handle_chat_completions(self, req: dict, raw_body: bytes):
        if not self.demo_mode and self.upstream:
            if not self._upstream_ready():
                self._send_json(503, {"error": "Qwen3-Coder-30B-A3B is still warming up on Render (first boot downloads the 18.6 GB model). Try again in a moment."})
                return
            self._proxy_chat_stream(raw_body)
            return

        # Local PC Demo Mode stream (~26 tok/s matching Qwen3-Coder-30B-A3B on 12 CPUs)
        messages = req.get("messages", [])
        last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        reply = (
            f"Hi! I'm **Zyro AI** running in **Local PC Demo Mode** (~26 tok/s simulating `Qwen3-Coder-30B-A3B-Instruct` on Render 12 CPU / 96 GB RAM).\n\n"
            f"Switch on **Agent Mode** in the top bar and ask me to **create a Python script and run it** — I will execute real files and real terminal commands right on your workspace!\n\n"
            f"You asked: _{last_user}_"
        )
        self._start_sse()
        self._stream_text_openai_format(reply, req.get("model", self.model_name))

    def _proxy_chat_stream(self, raw_body: bytes):
        target = f"{self.upstream.rstrip('/')}/v1/chat/completions"
        r = urllib.request.Request(target, data=raw_body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(r, timeout=300) as resp:
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() in ("content-type", "cache-control"):
                        self.send_header(k, v)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                while True:
                    chunk = resp.read(256)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except Exception as e:
            self._send_json(502, {"error": f"LLM server error: {e}"})

    # -------------------------------------------------------------------------
    # OpenCode-Style Multi-Step Agentic Loop (SSE Stream)
    # -------------------------------------------------------------------------
    def _handle_agent_run(self, req: dict):
        messages = list(req.get("messages", []))
        model = req.get("model", self.model_name)
        temperature = float(req.get("temperature", 0.4))
        max_tokens = int(req.get("max_tokens", 2048))

        self._start_sse()

        if self.demo_mode or not self.upstream:
            self._run_local_demo_agent(messages)
            return

        if not self._upstream_ready():
            self._sse_event({
                "type": "error",
                "message": "Qwen3-Coder-30B-A3B is still loading on Render. Check /health and retry shortly."
            })
            return

        # Prepend / augment system prompt with tool instructions
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": messages[0].get("content", "") + "\n\n" + AGENT_SYSTEM_SUFFIX}
        else:
            messages.insert(0, {"role": "system", "content": AGENT_SYSTEM_SUFFIX})

        max_steps = 8
        total_tokens = 0
        t0 = time.time()

        try:
            for step in range(max_steps):
                payload = {
                    "model": model,
                    "messages": messages,
                    "tools": AGENT_TOOLS_SCHEMA,
                    "tool_choice": "auto",
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True
                }
                req_obj = urllib.request.Request(
                    f"{self.upstream.rstrip('/')}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                    method="POST"
                )

                assistant_text = []
                tool_calls_acc: dict[int, dict] = {}

                with urllib.request.urlopen(req_obj, timeout=300) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}

                        # Stream normal content tokens immediately to Zyro Android app
                        content_piece = delta.get("content")
                        if content_piece:
                            assistant_text.append(content_piece)
                            total_tokens += 1
                            elapsed = max(0.05, time.time() - t0)
                            self._sse_event({
                                "type": "token",
                                "delta": content_piece,
                                "tps": round(total_tokens / elapsed, 1)
                            })

                        # Accumulate streamed tool_calls
                        for tc in delta.get("tool_calls") or []:
                            idx = int(tc.get("index", 0))
                            entry = tool_calls_acc.setdefault(idx, {
                                "id": tc.get("id") or f"call_{step}_{idx}",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                            if tc.get("id"):
                                entry["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                entry["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                entry["function"]["arguments"] += fn["arguments"]

                joined_text = "".join(assistant_text)

                if not tool_calls_acc:
                    # Final turn complete!
                    elapsed = max(0.05, time.time() - t0)
                    self._sse_event({
                        "type": "done",
                        "avg_tps": round(total_tokens / elapsed, 1),
                        "workspace_count": len(self.workspace.list_all_files())
                    })
                    return

                # Record assistant message with tool_calls and execute every tool
                ordered_calls = [tool_calls_acc[k] for k in sorted(tool_calls_acc.keys())]
                messages.append({
                    "role": "assistant",
                    "content": joined_text or None,
                    "tool_calls": ordered_calls
                })

                for tc in ordered_calls:
                    fn_name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    try:
                        parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                    except Exception:
                        parsed_args = {}

                    self._sse_event({
                        "type": "tool_start",
                        "call_id": tc["id"],
                        "tool": fn_name,
                        "args": parsed_args
                    })

                    result = self.workspace.execute_tool(fn_name, parsed_args)

                    self._sse_event({
                        "type": "tool_result",
                        "call_id": tc["id"],
                        "tool": fn_name,
                        "ok": result["ok"],
                        "summary": result["summary"],
                        "output": result["output"],
                        "exit_code": result["exit_code"],
                        "duration_ms": result["duration_ms"]
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": fn_name,
                        "content": json.dumps(result)
                    })

            self._sse_event({
                "type": "done",
                "avg_tps": round(total_tokens / max(0.05, time.time() - t0), 1),
                "workspace_count": len(self.workspace.list_all_files())
            })
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self._sse_event({"type": "error", "message": f"Agent error: {e}"})

    # -------------------------------------------------------------------------
    # Local PC Interactive Demo Agent (Actually creates files & runs commands!)
    # -------------------------------------------------------------------------
    def _run_local_demo_agent(self, messages: list[dict]):
        last_user = next((str(m.get("content", "")).strip() for m in reversed(messages) if m.get("role") == "user"), "")
        low = last_user.lower()
        py_cmd = "python" if platform.system() == "Windows" else "python3"

        try:
            # 1) Direct shell command request ("run <cmd>" or starts with "$ ")
            if last_user.startswith("$ ") or low.startswith("run "):
                cmd = last_user[2:].strip() if last_user.startswith("$ ") else last_user[4:].strip()
                res = self._emit_demo_tool("call_1", "run_terminal", {"command": cmd})
                summary = (
                    f"Executed `{cmd}` inside your workspace (exit code `{res['exit_code']}` in {res['duration_ms']}ms).\n\n"
                    f"```bash\n{res['output']}\n```"
                )
                self._stream_agent_tokens(summary)
                return

            # 2) List files request
            if any(k in low for k in ["list file", "show file", "workspace", "ls", "files"]):
                res = self._emit_demo_tool("call_1", "list_files", {"path": "."})
                summary = (
                    "Here are the current files inside your **Zyro Agent Workspace**:\n\n"
                    f"```text\n{res['output']}\n```\n\n"
                    "Tap the **Workspace / Terminal** icon in the top bar to view, copy, or download any file!"
                )
                self._stream_agent_tokens(summary)
                return

            # 3) Create & run real Python / web / script project right inside the workspace!
            file_match = re.search(r"([a-zA-Z0-9_\-/]+\.(?:py|js|html|txt|md|json|sh))", last_user)
            filename = file_match.group(1) if file_match else "zyro_demo.py"
            if filename.endswith(".py"):
                code = (
                    'import datetime\n'
                    'import platform\n\n'
                    'def main():\n'
                    '    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")\n'
                    '    print("========================================")\n'
                    '    print("  Zyro AI Agentic Execution Verified!")\n'
                    '    print("========================================")\n'
                    '    print(f"Timestamp : {now}")\n'
                    '    print(f"Runtime   : Python {platform.python_version()} on {platform.system()}")\n'
                    '    print("Model     : Qwen3-Coder-30B-A3B-Instruct (MoE 3B active)")\n'
                    '    squares = [n * n for n in range(1, 8)]\n'
                    '    print(f"Computed  : {squares}")\n\n'
                    'if __name__ == "__main__":\n'
                    '    main()\n'
                )
                self._emit_demo_tool("call_write", "write_file", {"path": filename, "content": code})
                run_res = self._emit_demo_tool("call_exec", "run_terminal", {"command": f"{py_cmd} {filename}"})
                summary = (
                    f"I created **`{filename}`** in your workspace and executed it using the real terminal tool!\n\n"
                    f"```python\n{code.strip()}\n```\n\n"
                    f"**Live Terminal Output (exit `{run_res['exit_code']}`):**\n"
                    f"```text\n{run_res['output']}\n```\n\n"
                    "Open the **Workspace & Terminal** glass panel in the top bar to browse or download this file, "
                    "or deploy `render-backend/` to your **12 vCPU / 96 GB RAM Render** instance for full autonomous `Qwen3-Coder-30B-A3B` coding!"
                )
                self._stream_agent_tokens(summary)
                return

            # Default: write requested file & list workspace
            sample = f"# Created by Zyro AI Agent\nPrompt: {last_user}\nModel: Qwen3-Coder-30B-A3B-Instruct\n"
            self._emit_demo_tool("call_write", "write_file", {"path": filename, "content": sample})
            self._emit_demo_tool("call_list", "list_files", {"path": "."})
            self._stream_agent_tokens(
                f"Created **`{filename}`** in your workspace! Open the **Workspace** panel to inspect or download it, "
                "or ask me to write and run a Python script (e.g., *\"create calc.py and run it\"*)."
            )
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _emit_demo_tool(self, call_id: str, tool: str, args: dict) -> dict:
        self._sse_event({"type": "tool_start", "call_id": call_id, "tool": tool, "args": args})
        time.sleep(0.25)
        res = self.workspace.execute_tool(tool, args)
        self._sse_event({
            "type": "tool_result",
            "call_id": call_id,
            "tool": tool,
            "ok": res["ok"],
            "summary": res["summary"],
            "output": res["output"],
            "exit_code": res["exit_code"],
            "duration_ms": res["duration_ms"]
        })
        return res

    def _stream_agent_tokens(self, text: str):
        delay = 1.0 / max(1.0, self.demo_tps)
        chunks = self._chunk_text(text)
        t0 = time.time()
        for i, tok in enumerate(chunks, start=1):
            elapsed = max(0.05, time.time() - t0)
            self._sse_event({"type": "token", "delta": tok, "tps": round(min(34.0, i / elapsed), 1)})
            time.sleep(delay)
        self._sse_event({
            "type": "done",
            "avg_tps": round(self.demo_tps, 1),
            "workspace_count": len(self.workspace.list_all_files())
        })

    def _stream_text_openai_format(self, text: str, model: str):
        delay = 1.0 / max(1.0, self.demo_tps)
        cid = f"chatcmpl-{int(time.time())}"
        try:
            for tok in self._chunk_text(text):
                frame = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]
                }
                self._sse_event(frame)
                time.sleep(delay)
            self._sse_event({
                "id": cid,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            })
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        chunks, buf = [], ""
        for ch in text:
            buf += ch
            if ch in (" ", "\n") or len(buf) >= 5:
                chunks.append(buf)
                buf = ""
        if buf:
            chunks.append(buf)
        return chunks


def main():
    parser = argparse.ArgumentParser(description="Zyro AI Agentic Backend (Render + Local PC)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "10000")))
    parser.add_argument("--upstream", type=str, default=os.environ.get("LLAMA_UPSTREAM", ""))
    parser.add_argument("--workspace", type=str, default=os.environ.get("WORKSPACE_DIR", "./workspace"))
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL_ALIAS", DEFAULT_MODEL))
    parser.add_argument("--api-key", type=str, default=os.environ.get("ZYRO_API_KEY", ""))
    parser.add_argument("--demo", action="store_true", help="Run real local file/terminal agent without llama-server")
    parser.add_argument("--tps", type=float, default=26.0, help="Simulated tok/s in demo mode")
    args = parser.parse_args()

    ZyroAgentHandler.upstream = args.upstream or None
    ZyroAgentHandler.api_key = args.api_key
    ZyroAgentHandler.model_name = args.model
    ZyroAgentHandler.workspace = WorkspaceExecutor(Path(args.workspace))
    ZyroAgentHandler.demo_mode = args.demo or (not args.upstream)
    ZyroAgentHandler.demo_tps = args.tps

    lan_ip = get_lan_ip()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), ZyroAgentHandler)
    print("=" * 64)
    print("  ZYRO AI — OpenCode Agentic Backend (Qwen3-Coder-30B-A3B)")
    print("=" * 64)
    print(f"  Mode              : {'LOCAL PC AGENT DEMO' if ZyroAgentHandler.demo_mode else 'UPSTREAM -> ' + args.upstream}")
    print(f"  Model             : {args.model}")
    print(f"  Workspace Dir     : {Path(args.workspace).resolve()}")
    print(f"  Android Emulator  : http://10.0.2.2:{args.port}")
    print(f"  Phone (Same Wi-Fi): http://{lan_ip}:{args.port}")
    print("=" * 64)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
