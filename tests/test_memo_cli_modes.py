"""CLI tests for memo local-mode selection and explicit target resolution."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from mind_map.app.cli.main import app
from mind_map.rag.graph_store import GraphStore


runner = CliRunner()


def init_store(tmp_path: Path) -> None:
    store = GraphStore(tmp_path)
    store.initialize()


class TestMemoCliModes:
    def test_requires_minimax_api_key_when_no_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            init_store(Path(tmpdir))
            with patch.dict(os.environ, {}, clear=True):
                result = runner.invoke(app, ["memo", "hello world", "--data-dir", tmpdir])
                assert result.exit_code == 1
                assert "MINIMAX_API_KEY not set" in result.stdout

    def test_rejects_openclaw_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            init_store(Path(tmpdir))
            result = runner.invoke(
                app,
                ["memo", "hello world", "--data-dir", tmpdir, "--openclaw", "minimax"],
            )
            assert result.exit_code != 0
            assert "No such option: --openclaw" in result.stdout

    def test_help_lists_only_local_option_for_memo(self) -> None:
        result = runner.invoke(app, ["memo", "--help"])
        assert result.exit_code == 0
        assert "--local" in result.stdout
        assert "--openclaw" not in result.stdout

    def test_local_resolves_first_model_when_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            init_store(Path(tmpdir))
            with patch("mind_map.processor.cli_executor.resolve_local_model", return_value="mlx-community/gemma-4-e4b-it-4bit"):
                with patch("mind_map.app.pipeline.ingest_memo_cli", return_value=(True, "Created 1 nodes", ["n1"])) as mock_ingest:
                    result = runner.invoke(app, ["memo", "hello world long enough", "--data-dir", tmpdir, "--local", ""])
                    assert result.exit_code == 0
                    target = mock_ingest.call_args.kwargs["target"]
                    assert target.model == "mlx-community/gemma-4-e4b-it-4bit"
                    assert target.base_url == "http://127.0.0.1:11435/v1"

    def test_local_uses_explicit_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            init_store(Path(tmpdir))
            with patch("mind_map.app.pipeline.ingest_memo_cli", return_value=(True, "Created 1 nodes", ["n1"])) as mock_ingest:
                result = runner.invoke(
                    app,
                    ["memo", "hello world long enough", "--data-dir", tmpdir, "--local", "mlx-community/gemma-4-e4b-it-4bit"],
                )
                assert result.exit_code == 0
                target = mock_ingest.call_args.kwargs["target"]
                assert target.model == "mlx-community/gemma-4-e4b-it-4bit"
