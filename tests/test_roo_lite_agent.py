from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kv_compaction_qwen35_clean.roo_lite_agent import (
    RooLiteAgent,
    RooLiteToolExecutor,
    ToolExecutionResult,
)


class _FakeService:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(self, *, messages, tools, max_tokens: int, temperature: float, top_p: float):
        self.calls.append(
            {
                "messages": json.loads(json.dumps(messages)),
                "tools": json.loads(json.dumps(tools)),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        response = self.responses.pop(0)
        return response, SimpleNamespace()


def _tool_call_response(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "using tool",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def _final_response(content: str) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    }


def test_tool_executor_append_artifact_creates_and_appends(tmp_path: Path) -> None:
    executor = RooLiteToolExecutor(
        workspace_root=tmp_path,
        writable_root=Path("artifacts/roo_lite"),
        shell_approver=lambda command, workdir: True,
    )

    first = executor.run("append_artifact", {"path": "notes/run.txt", "content": "hello"})
    second = executor.run("append_artifact", {"path": "notes/run.txt", "content": "\nworld"})

    target = tmp_path / "artifacts" / "roo_lite" / "notes" / "run.txt"
    assert target.read_text(encoding="utf-8") == "hello\nworld"
    assert json.loads(first.content)["created"] is True
    assert json.loads(second.content)["created"] is False


def test_tool_executor_shell_denied_returns_status(tmp_path: Path) -> None:
    executor = RooLiteToolExecutor(
        workspace_root=tmp_path,
        writable_root=Path("artifacts/roo_lite"),
        shell_approver=lambda command, workdir: False,
    )

    result = executor.run("run_shell", {"command": "pwd"})

    payload = json.loads(result.content)
    assert payload["approved"] is False
    assert payload["status"] == "denied_by_user"


def test_tool_executor_shell_executes_when_approved(tmp_path: Path) -> None:
    executor = RooLiteToolExecutor(
        workspace_root=tmp_path,
        writable_root=Path("artifacts/roo_lite"),
        shell_approver=lambda command, workdir: True,
    )

    result = executor.run("run_shell", {"command": "printf hello"})

    payload = json.loads(result.content)
    assert payload["approved"] is True
    assert payload["returncode"] == 0
    assert payload["stdout"] == "hello"


def test_roo_lite_agent_executes_tool_then_returns_final_text(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("first line\nsecond line\n", encoding="utf-8")
    service = _FakeService(
        [
            _tool_call_response("read_file", {"path": "README.md"}),
            _final_response("repo summary"),
        ]
    )
    agent = RooLiteAgent(
        service=service,
        workspace_root=tmp_path,
        writable_root=Path("artifacts/roo_lite"),
        shell_approver=lambda command, workdir: True,
        system_prompt="Be concise.",
        max_steps=4,
    )

    result = agent.run_turn("Tell me about the repo.")

    assert result.final_content == "repo summary"
    assert len(result.steps) == 2
    assert result.steps[0].tool_calls[0]["function"]["name"] == "read_file"
    assert "README.md" in result.steps[0].tool_results[0].content
    second_call_messages = service.calls[1]["messages"]
    assert second_call_messages[-1]["role"] == "tool"
    assert "first line" in str(second_call_messages[-1]["content"])


def test_tool_executor_rejects_writes_outside_writable_root(tmp_path: Path) -> None:
    executor = RooLiteToolExecutor(
        workspace_root=tmp_path,
        writable_root=Path("artifacts/roo_lite"),
        shell_approver=lambda command, workdir: True,
    )

    try:
        executor.run("append_artifact", {"path": "../escape.txt", "content": "bad"})
    except ValueError as exc:
        assert "escapes workspace" in str(exc)
    else:
        raise AssertionError("append_artifact should reject paths outside the writable root")


def test_search_files_ignores_virtualenv_by_default(tmp_path: Path) -> None:
    repo_file = tmp_path / "src" / "app.py"
    repo_file.parent.mkdir(parents=True, exist_ok=True)
    repo_file.write_text("from pathlib import Path\n", encoding="utf-8")
    venv_file = tmp_path / ".venv-test" / "lib.py"
    venv_file.parent.mkdir(parents=True, exist_ok=True)
    venv_file.write_text("from pathlib import Path\n", encoding="utf-8")
    executor = RooLiteToolExecutor(
        workspace_root=tmp_path,
        writable_root=Path("artifacts/roo_lite"),
        shell_approver=lambda command, workdir: True,
    )

    result = executor.run(
        "search_files",
        {
            "path": ".",
            "regex": "from pathlib import Path",
        },
    )

    payload = json.loads(result.content)
    assert [row["path"] for row in payload] == ["src/app.py"]


def test_append_artifact_normalizes_workspace_prefixed_target(tmp_path: Path) -> None:
    executor = RooLiteToolExecutor(
        workspace_root=tmp_path,
        writable_root=Path("artifacts/roo_lite"),
        shell_approver=lambda command, workdir: True,
    )

    executor.run(
        "append_artifact",
        {
            "path": "artifacts/roo_lite/notes/run.txt",
            "content": "hello\n",
        },
    )

    assert (tmp_path / "artifacts" / "roo_lite" / "notes" / "run.txt").read_text(encoding="utf-8") == "hello\n"


def test_roo_lite_agent_enforces_append_artifact_before_finalizing(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("repo thesis\n", encoding="utf-8")
    service = _FakeService(
        [
            _tool_call_response("read_file", {"path": "README.md"}),
            _final_response("I wrote the file."),
            _tool_call_response(
                "append_artifact",
                {"path": "notes/out.md", "content": "repo thesis\n"},
            ),
            _final_response("Wrote artifacts/roo_lite/notes/out.md"),
        ]
    )
    agent = RooLiteAgent(
        service=service,
        workspace_root=tmp_path,
        writable_root=Path("artifacts/roo_lite"),
        shell_approver=lambda command, workdir: True,
        system_prompt="Be concise.",
        max_steps=6,
    )

    result = agent.run_turn("Read README.md, then create artifacts/roo_lite/notes/out.md with one line and tell me the file you wrote.")

    assert result.final_content == "Wrote artifacts/roo_lite/notes/out.md"
    assert (tmp_path / "artifacts" / "roo_lite" / "notes" / "out.md").read_text(encoding="utf-8") == "repo thesis\n"
    reminder_call_messages = service.calls[2]["messages"]
    assert reminder_call_messages[-1]["role"] == "user"
    assert "append_artifact" in reminder_call_messages[-1]["content"]


def test_roo_lite_agent_does_not_return_stale_final_after_last_tool_step(tmp_path: Path) -> None:
    service = _FakeService(
        [
            _tool_call_response("read_file", {"path": "README.md"}),
            _final_response("I wrote the file."),
            _tool_call_response(
                "append_artifact",
                {"path": "notes/out.md", "content": "repo thesis\n"},
            ),
        ]
    )
    (tmp_path / "README.md").write_text("repo thesis\n", encoding="utf-8")
    agent = RooLiteAgent(
        service=service,
        workspace_root=tmp_path,
        writable_root=Path("artifacts/roo_lite"),
        shell_approver=lambda command, workdir: True,
        max_steps=3,
    )

    result = agent.run_turn("Read README.md, then create artifacts/roo_lite/notes/out.md with one line.")

    assert result.final_content is None
    assert result.steps[-1].tool_calls[0]["function"]["name"] == "append_artifact"
