from pathlib import Path


def test_qwen35_clean_repo_shape() -> None:
    assert Path("README.md").exists()
    assert Path("pyproject.toml").exists()
    assert Path("docs/plan.md").exists()
    assert Path("docs/repo_contract.md").exists()
    assert Path("docs/roo_proxy_experiment.md").exists()
    assert Path("configs/README.md").exists()
    assert Path("data/README.md").exists()
    assert Path("src/kv_compaction_qwen35_clean/roadmap.py").exists()
    assert Path("src/kv_compaction_qwen35_clean/qwen35_openai_proxy.py").exists()
    assert Path("src/kv_compaction_qwen35_clean/roo_lite_agent.py").exists()
