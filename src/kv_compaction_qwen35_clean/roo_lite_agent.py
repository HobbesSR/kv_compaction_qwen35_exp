from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable
import urllib.error
import urllib.request

from kv_compaction_qwen35_clean.qwen35_openai_proxy import Qwen35OpenAIProxyService


DEFAULT_AGENT_MAX_TOKENS = 1024
DEFAULT_AGENT_TEMPERATURE = 0.0
DEFAULT_AGENT_TOP_P = 1.0
DEFAULT_AGENT_MAX_STEPS = 8
DEFAULT_READ_LIMIT = 200
DEFAULT_SEARCH_LIMIT = 200
DEFAULT_SHELL_TIMEOUT_SECONDS = 30
DEFAULT_WRITABLE_ROOT = Path("artifacts/roo_lite")
DEFAULT_IGNORED_TOP_LEVEL_DIRS = (
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
)
DEFAULT_IGNORED_TOP_LEVEL_PREFIXES = (
    ".venv",
)


def build_default_agent_system_prompt(*, writable_root: Path) -> str:
    return (
        "You are a small local coding assistant. Use the provided tools when they help. "
        "You may answer directly in plain text once you have enough information. "
        "Prefer repository files under src/, tests/, docs/, scripts/, and configs/. "
        "Avoid hidden directories, virtual environments, and dependency caches unless the user explicitly asks for them. "
        "Prefer read-only inspection tools before using the shell. "
        f"The append_artifact tool may only create or append files under {writable_root.as_posix()}. "
        "If the user asks you to create, write, or append a file, you must use append_artifact before claiming success. "
        "Do not invent file contents or command results."
    )


def build_agent_tools(*, writable_root: Path) -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files under a repository-relative directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                    },
                    "required": ["path", "recursive"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file with line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search files recursively with a regular expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "regex": {"type": "string"},
                        "file_pattern": {"type": "string"},
                    },
                    "required": ["path", "regex"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Run a shell command in the repository after explicit user approval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "workdir": {"type": "string"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "append_artifact",
                "description": (
                    "Create or append text to a file under "
                    f"{writable_root.as_posix()}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _resolve_repo_path(root: Path, relative_path: str, *, allow_missing: bool = False) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {relative_path!r}") from exc
    if not allow_missing and not candidate.exists():
        raise FileNotFoundError(relative_path)
    return candidate


def _is_ignored_workspace_relative(workspace_relative: Path) -> bool:
    if not workspace_relative.parts:
        return False
    first = workspace_relative.parts[0]
    if first in DEFAULT_IGNORED_TOP_LEVEL_DIRS:
        return True
    return any(first.startswith(prefix) for prefix in DEFAULT_IGNORED_TOP_LEVEL_PREFIXES)


def _user_text_requires_append_artifact(user_text: str) -> bool:
    lowered = user_text.lower()
    has_write_verb = any(token in lowered for token in ("write ", "append ", "create "))
    mentions_file = "file" in lowered or "artifacts/" in lowered or ".md" in lowered or ".txt" in lowered
    return has_write_verb and mentions_file


def _format_numbered_lines(path: Path, *, offset: int, limit: int) -> str:
    if offset < 1:
        raise ValueError("offset must be >= 1")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    lines = path.read_text(encoding="utf-8").splitlines()
    start = offset - 1
    selected = lines[start : start + limit]
    rendered = [f"{index:4d} | {line}" for index, line in enumerate(selected, start=offset)]
    return f"File: {path.relative_to(path.parents[len(path.parts) - 1])}\n" + "\n".join(rendered)


def _format_numbered_lines_for_display(display_path: str, text: str, *, offset: int, limit: int) -> str:
    if offset < 1:
        raise ValueError("offset must be >= 1")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    lines = text.splitlines()
    selected = lines[offset - 1 : offset - 1 + limit]
    rendered = [f"{index:4d} | {line}" for index, line in enumerate(selected, start=offset)]
    return f"File: {display_path}\n" + "\n".join(rendered)


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    content: str


class RooLiteToolExecutor:
    def __init__(
        self,
        *,
        workspace_root: Path,
        writable_root: Path,
        shell_approver: Callable[[str, str], bool],
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.writable_root = _resolve_repo_path(
            self.workspace_root,
            writable_root.as_posix(),
            allow_missing=True,
        )
        self.writable_root.mkdir(parents=True, exist_ok=True)
        self.shell_approver = shell_approver

    def _resolve_writable_target(self, relative_path: str) -> Path:
        requested = Path(relative_path)
        try:
            workspace_relative = requested.relative_to(self.writable_root.relative_to(self.workspace_root))
        except ValueError:
            workspace_relative = requested
        return _resolve_repo_path(self.writable_root, workspace_relative.as_posix(), allow_missing=True)

    def run(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        handler = getattr(self, f"_run_{tool_name}", None)
        if handler is None:
            raise ValueError(f"Unsupported tool {tool_name!r}")
        content = handler(arguments)
        return ToolExecutionResult(tool_name=tool_name, content=content)

    def _run_list_files(self, arguments: dict[str, Any]) -> str:
        relative = str(arguments.get("path", "."))
        recursive = bool(arguments.get("recursive", False))
        target = _resolve_repo_path(self.workspace_root, relative)
        if not target.is_dir():
            raise ValueError(f"Not a directory: {relative!r}")
        if recursive:
            entries = sorted(
                path.relative_to(self.workspace_root).as_posix()
                for path in target.rglob("*")
            )
        else:
            entries = sorted(
                path.relative_to(self.workspace_root).as_posix()
                for path in target.iterdir()
            )
        return json.dumps({"path": relative, "recursive": recursive, "entries": entries}, indent=2)

    def _run_read_file(self, arguments: dict[str, Any]) -> str:
        relative = str(arguments["path"])
        offset = int(arguments.get("offset", 1))
        limit = int(arguments.get("limit", DEFAULT_READ_LIMIT))
        target = _resolve_repo_path(self.workspace_root, relative)
        if not target.is_file():
            raise ValueError(f"Not a file: {relative!r}")
        text = target.read_text(encoding="utf-8")
        return _format_numbered_lines_for_display(relative, text, offset=offset, limit=limit)

    def _run_search_files(self, arguments: dict[str, Any]) -> str:
        relative = str(arguments.get("path", "."))
        pattern = str(arguments["regex"])
        file_pattern = str(arguments.get("file_pattern", "*"))
        target = _resolve_repo_path(self.workspace_root, relative)
        if not target.is_dir():
            raise ValueError(f"Not a directory: {relative!r}")
        target_workspace_relative = target.relative_to(self.workspace_root)
        disable_ignored_filter = _is_ignored_workspace_relative(target_workspace_relative)
        regex = re.compile(pattern)
        rows: list[dict[str, Any]] = []
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            workspace_relative = path.relative_to(self.workspace_root)
            if not disable_ignored_filter and _is_ignored_workspace_relative(workspace_relative):
                continue
            if not fnmatch.fnmatch(path.name, file_pattern):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for index, line in enumerate(lines, start=1):
                if regex.search(line):
                    rows.append(
                        {
                            "path": path.relative_to(self.workspace_root).as_posix(),
                            "line": index,
                            "text": line,
                        }
                    )
                    if len(rows) >= DEFAULT_SEARCH_LIMIT:
                        return json.dumps(rows, indent=2)
        return json.dumps(rows, indent=2)

    def _run_run_shell(self, arguments: dict[str, Any]) -> str:
        command = str(arguments["command"])
        workdir = str(arguments.get("workdir", "."))
        cwd = _resolve_repo_path(self.workspace_root, workdir)
        if not cwd.is_dir():
            raise ValueError(f"Not a directory: {workdir!r}")
        approved = self.shell_approver(command, cwd.relative_to(self.workspace_root).as_posix())
        if not approved:
            return json.dumps(
                {
                    "approved": False,
                    "command": command,
                    "workdir": cwd.relative_to(self.workspace_root).as_posix(),
                    "status": "denied_by_user",
                },
                indent=2,
            )
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=DEFAULT_SHELL_TIMEOUT_SECONDS,
        )
        return json.dumps(
            {
                "approved": True,
                "command": command,
                "workdir": cwd.relative_to(self.workspace_root).as_posix(),
                "returncode": completed.returncode,
                "stdout": completed.stdout[-12000:],
                "stderr": completed.stderr[-12000:],
            },
            indent=2,
        )

    def _run_append_artifact(self, arguments: dict[str, Any]) -> str:
        relative = str(arguments["path"])
        content = str(arguments["content"])
        target = self._resolve_writable_target(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return json.dumps(
            {
                "path": target.relative_to(self.workspace_root).as_posix(),
                "created": not existed,
                "appended_chars": len(content),
            },
            indent=2,
        )


def interactive_shell_approver(command: str, workdir: str) -> bool:
    prompt = (
        "\nShell tool requested\n"
        f"  workdir: {workdir}\n"
        f"  command: {command}\n"
        "Approve? [y/N]: "
    )
    return input(prompt).strip().lower() in {"y", "yes"}


@dataclass(frozen=True)
class AgentStep:
    assistant_content: str | None
    tool_calls: list[dict[str, object]]
    tool_results: list[ToolExecutionResult]
    finish_reason: str


@dataclass(frozen=True)
class AgentTurnResult:
    final_content: str | None
    steps: list[AgentStep]
    messages: list[dict[str, object]]


class RooLiteAgent:
    def __init__(
        self,
        *,
        service: Qwen35OpenAIProxyService,
        workspace_root: Path,
        writable_root: Path = DEFAULT_WRITABLE_ROOT,
        shell_approver: Callable[[str, str], bool] = interactive_shell_approver,
        system_prompt: str | None = None,
        max_tokens: int = DEFAULT_AGENT_MAX_TOKENS,
        temperature: float = DEFAULT_AGENT_TEMPERATURE,
        top_p: float = DEFAULT_AGENT_TOP_P,
        max_steps: int = DEFAULT_AGENT_MAX_STEPS,
    ) -> None:
        self.service = service
        self.workspace_root = workspace_root.resolve()
        self.writable_root = writable_root
        self.tool_executor = RooLiteToolExecutor(
            workspace_root=self.workspace_root,
            writable_root=writable_root,
            shell_approver=shell_approver,
        )
        self.tools = build_agent_tools(writable_root=writable_root)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_steps = max_steps
        self.messages: list[dict[str, object]] = []
        system_text = system_prompt or build_default_agent_system_prompt(writable_root=writable_root)
        self.messages.append({"role": "system", "content": system_text})

    def run_turn(self, user_text: str) -> AgentTurnResult:
        self.messages.append({"role": "user", "content": user_text})
        steps: list[AgentStep] = []
        final_content: str | None = None
        write_required = _user_text_requires_append_artifact(user_text)
        append_completed = False

        for _ in range(self.max_steps):
            payload, _metrics = self.service.complete(
                messages=self.messages,
                tools=self.tools,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            choice = dict(payload["choices"][0])
            message = dict(choice.get("message", {}))
            assistant_content = message.get("content")
            if assistant_content is not None:
                assistant_content = str(assistant_content)
            tool_calls = list(message.get("tool_calls", []) or [])
            finish_reason = str(choice.get("finish_reason", "stop"))
            assistant_record: dict[str, object] = {"role": "assistant", "content": assistant_content}
            if tool_calls:
                assistant_record["tool_calls"] = tool_calls
            self.messages.append(assistant_record)

            tool_results: list[ToolExecutionResult] = []
            if tool_calls:
                for tool_call in tool_calls:
                    function_block = dict(tool_call.get("function", {}))
                    tool_name = str(function_block.get("name"))
                    raw_arguments = function_block.get("arguments", "{}")
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
                    result = self.tool_executor.run(tool_name, arguments)
                    if tool_name == "append_artifact":
                        append_completed = True
                    tool_results.append(result)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(tool_call.get("id")),
                            "content": result.content,
                        }
                    )
                steps.append(
                    AgentStep(
                        assistant_content=assistant_content,
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                        finish_reason=finish_reason,
                    )
                )
                continue

            if write_required and not append_completed:
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have not completed the requested file write yet. "
                            "Use append_artifact to create or append the requested file before giving a final answer."
                        ),
                    }
                )
                steps.append(
                    AgentStep(
                        assistant_content=assistant_content,
                        tool_calls=[],
                        tool_results=[],
                        finish_reason=finish_reason,
                    )
                )
                continue

            final_content = assistant_content
            steps.append(
                AgentStep(
                    assistant_content=assistant_content,
                    tool_calls=[],
                    tool_results=[],
                    finish_reason=finish_reason,
                )
            )
            break

        if final_content is None and steps:
            if steps[-1].tool_calls:
                final_content = None
            else:
                final_content = steps[-1].assistant_content
        return AgentTurnResult(
            final_content=final_content,
            steps=steps,
            messages=list(self.messages),
        )


class OpenAIProxyClientService:
    def __init__(self, *, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Proxy request failed: HTTP {exc.code}: {details}") from exc
        metrics = body.get("kv_compaction", {})
        return body, metrics

    def close(self) -> None:
        return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small local Roo-lite agent loop.")
    parser.add_argument(
        "--config",
        default="configs/qwen35_smoke/qwen3_5_9b.yaml",
        help="Experiment config used to load the Qwen model.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional existing OpenAI-compatible proxy base URL, for example http://127.0.0.1:8010/v1.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Model name used with --base-url.",
    )
    parser.add_argument(
        "--cache-root",
        default="artifacts/qwen35_proxy_cache",
        help="Directory for the proxy/service cache root.",
    )
    parser.add_argument(
        "--writable-root",
        default=str(DEFAULT_WRITABLE_ROOT),
        help="Repository-relative directory where append_artifact may write.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt override.",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Optional one-shot user message. If omitted, runs an interactive loop.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_AGENT_MAX_TOKENS,
        help="Max generation tokens per assistant step.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_AGENT_MAX_STEPS,
        help="Max model/tool steps per user turn.",
    )
    parser.add_argument(
        "--shell-approval",
        choices=("ask", "always", "never"),
        default="ask",
        help="Shell tool approval policy.",
    )
    return parser


def resolve_shell_approver(policy: str) -> Callable[[str, str], bool]:
    if policy == "always":
        return lambda command, workdir: True
    if policy == "never":
        return lambda command, workdir: False
    return interactive_shell_approver


def run_cli(args: argparse.Namespace) -> int:
    if args.base_url:
        service = OpenAIProxyClientService(
            base_url=args.base_url,
            model=args.model,
        )
    else:
        service = Qwen35OpenAIProxyService(
            config_path=args.config,
            cache_root=args.cache_root,
        )
    agent = RooLiteAgent(
        service=service,
        workspace_root=Path.cwd(),
        writable_root=Path(args.writable_root),
        shell_approver=resolve_shell_approver(args.shell_approval),
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
    )
    try:
        if args.message is not None:
            result = agent.run_turn(args.message)
            print(result.final_content or "")
            return 0

        while True:
            try:
                user_text = input("\nuser> ").strip()
            except EOFError:
                print()
                return 0
            if not user_text:
                continue
            if user_text in {"/quit", "/exit"}:
                return 0
            result = agent.run_turn(user_text)
            print(f"\nassistant> {result.final_content or ''}")
    finally:
        service.close()


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
