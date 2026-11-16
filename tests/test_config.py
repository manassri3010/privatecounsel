from app.config import Settings


def test_reads_dotenv_and_ignores_unrelated_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=qwen2.5:3b\nPOSTGRES_USER=privatecounsel\n")
    assert Settings(_env_file=env).llm_model == "qwen2.5:3b"


def test_environment_variable_overrides_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=qwen2.5:3b\n")
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    assert Settings(_env_file=env).llm_model == "Qwen/Qwen2.5-7B-Instruct"
