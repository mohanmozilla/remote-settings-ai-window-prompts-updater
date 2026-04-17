import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from kinto_http import KintoException

from ai_window_prompts_updater import (
    clone_repo,
    collect_prompts_and_params,
    fetch_current_prompts,
    get_item,
    main,
    sync_collection,
)


@pytest.fixture
def mocked_client():
    # Mock the kinto_http.Client where it is imported.
    with mock.patch("ai_window_prompts_updater.Client", spec=True) as mocked_class:
        yield mocked_class()


@pytest.fixture
def temp_prompts_dir():
    """Create a temporary directory structure with sample prompts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        prompts_dir = Path(temp_dir) / "prompts"
        feature_dir = prompts_dir / "chat"
        version_dir = feature_dir / "v1"
        version_dir.mkdir(parents=True)

        # Create sample JSON file
        json_data = {
            "feature": "chat",
            "model": "claude.3.5",
            "parameters": {"temperature": 0.7, "max_tokens": 1000},
        }
        with open(version_dir / "claude.3.5.json", "w") as f:
            json.dump(json_data, f)

        # Create sample MD file
        with open(version_dir / "claude.3.5.md", "w") as f:
            f.write("You are a helpful assistant.")

        yield prompts_dir


def test_cannot_call_unknown_method(mocked_client):
    # Thanks to the spec=True argument, the mock will raise an
    # AttributeError if we try to call a method that doesn't exist
    with pytest.raises(AttributeError):
        mocked_client.unknown_method()


def test_main_anonymous(mocked_client, capsys):
    mocked_client.server_info.return_value = {}

    main()

    mocked_client.server_info.assert_called_once()
    assert "⚠️ Anonymous" in capsys.readouterr().out


def test_main_logged_in(mocked_client, capsys):
    mocked_client.server_info.return_value = {"user": {"id": "account:bot"}}

    main()

    mocked_client.server_info.assert_called_once()
    assert "Logged in as account:bot" in capsys.readouterr().out


# Tests for clone_repo function
@pytest.mark.parametrize(
    "env,expected_branch",
    [
        ("prod", "prod"),
        ("dev", "prod"),
        ("stage", "stage"),
    ],
)
@mock.patch("ai_window_prompts_updater.GIT_TOKEN", "test_token")
@mock.patch("ai_window_prompts_updater.PROMPTS_REPO", "https://github.com/test/repo.git")
@mock.patch("ai_window_prompts_updater.subprocess.run")
def test_clone_repo_success(mock_run, capsys, env, expected_branch):
    mock_run.return_value = mock.Mock(returncode=0, stderr="")

    result, branch_name = clone_repo(env)

    assert result != ""
    assert branch_name == expected_branch
    assert "ai-window-remote-settings-prompts" in str(result)
    assert "✅ Repository cloned successfully" in capsys.readouterr().out
    mock_run.assert_called_once()


@mock.patch("ai_window_prompts_updater.GIT_TOKEN", "test_token")
@mock.patch("ai_window_prompts_updater.PROMPTS_REPO", "https://github.com/test/repo.git")
@mock.patch("ai_window_prompts_updater.subprocess.run")
def test_clone_repo_failure(mock_run, capsys):
    mock_run.return_value = mock.Mock(returncode=1, stderr="Authentication failed")

    result, _ = clone_repo("prod")

    assert result == ""
    output = capsys.readouterr().out
    assert "ERROR cloning repo" in output


@mock.patch("ai_window_prompts_updater.GIT_TOKEN", "test_token")
@mock.patch("ai_window_prompts_updater.PROMPTS_REPO", "https://github.com/test/repo.git")
@mock.patch("ai_window_prompts_updater.subprocess.run")
def test_clone_repo_with_token(mock_run):
    mock_run.return_value = mock.Mock(returncode=0, stderr="")

    clone_repo("prod")

    # Verify token was inserted into URL
    call_args = mock_run.call_args[0][0]
    assert "https://test_token@github.com/test/repo.git" in call_args


# Tests for get_item function
def test_get_item(temp_prompts_dir):
    version_dir = temp_prompts_dir / "chat" / "v1"

    result = get_item(version_dir, "claude.3.5")

    assert result["id"] == "chat--claude-3-5--v1"
    assert result["feature"] == "chat"
    assert result["model"] == "claude.3.5"
    assert result["prompts"] == "You are a helpful assistant."
    assert isinstance(result["parameters"], str)
    params = json.loads(result["parameters"])
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 1000


# Tests for collect_prompts_and_params function
def test_collect_prompts_and_params(temp_prompts_dir):
    records = collect_prompts_and_params(temp_prompts_dir)

    assert len(records) == 1
    assert records[0]["id"] == "chat--claude-3-5--v1"
    assert records[0]["feature"] == "chat"


def test_collect_prompts_and_params_multiple_versions(temp_prompts_dir):
    # Add another version
    v2_dir = temp_prompts_dir / "chat" / "v2"
    v2_dir.mkdir()

    json_data = {
        "feature": "chat",
        "model": "claude.3.5",
        "parameters": {"temperature": 0.5},
    }
    with open(v2_dir / "claude.3.5.json", "w") as f:
        json.dump(json_data, f)
    with open(v2_dir / "claude.3.5.md", "w") as f:
        f.write("Updated prompt")

    records = collect_prompts_and_params(temp_prompts_dir)

    assert len(records) == 2
    ids = [r["id"] for r in records]
    assert "chat--claude-3-5--v1" in ids
    assert "chat--claude-3-5--v2" in ids


# Tests for fetch_current_prompts function
@mock.patch("ai_window_prompts_updater.shutil.rmtree")
@mock.patch("ai_window_prompts_updater.collect_prompts_and_params")
def test_fetch_current_prompts(mock_collect, mock_rmtree, temp_prompts_dir, capsys):
    mock_collect.return_value = [{"id": "test-1"}, {"id": "test-2"}]
    repo_path = temp_prompts_dir.parent

    records = fetch_current_prompts(repo_path)

    assert len(records) == 2
    mock_collect.assert_called_once()
    mock_rmtree.assert_called_once()
    output = capsys.readouterr().out
    assert "📦 Found 2 prompt records" in output
    assert "🧹 Cleaned up temporary directory" in output


def test_fetch_current_prompts_missing_directory(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)

        with pytest.raises(FileNotFoundError, match="Prompts directory not found"):
            fetch_current_prompts(repo_path)


# Tests for sync_collection function
def test_sync_collection_no_changes(capsys):
    mock_client = mock.Mock()
    mock_client.get_records.return_value = [{"id": "test-1", "data": "foo"}]
    source_records = [{"id": "test-1", "data": "foo"}]

    result = sync_collection(mock_client, source_records)

    assert result == 0
    output = capsys.readouterr().out
    assert "Records are already in sync" in output
    mock_client.request_review.assert_not_called()


def test_sync_collection_with_creates():
    mock_client = mock.Mock()
    mock_client.get_records.return_value = []
    mock_batch = mock.Mock()
    mock_batch.results.return_value = [{}]
    mock_client.batch.return_value.__enter__ = mock.Mock(return_value=mock_batch)
    mock_client.batch.return_value.__exit__ = mock.Mock(return_value=False)

    source_records = [{"id": "test-1", "data": "foo"}]

    result = sync_collection(mock_client, source_records)

    assert result == 0
    mock_batch.create_record.assert_called_once()
    mock_client.request_review.assert_called_once()


def test_sync_collection_with_updates():
    mock_client = mock.Mock()
    mock_client.get_records.return_value = [{"id": "test-1", "data": "old", "last_modified": 123}]
    mock_batch = mock.Mock()
    mock_batch.results.return_value = [{}]
    mock_client.batch.return_value.__enter__ = mock.Mock(return_value=mock_batch)
    mock_client.batch.return_value.__exit__ = mock.Mock(return_value=False)

    source_records = [{"id": "test-1", "data": "new"}]

    result = sync_collection(mock_client, source_records)

    assert result == 0
    mock_batch.update_record.assert_called_once()
    # Verify last_modified was removed
    call_args = mock_batch.update_record.call_args
    assert "last_modified" not in call_args.kwargs["data"]


def test_sync_collection_with_deletes():
    mock_client = mock.Mock()
    mock_client.get_records.return_value = [{"id": "test-1", "data": "foo"}]
    mock_batch = mock.Mock()
    mock_batch.results.return_value = [{}]
    mock_client.batch.return_value.__enter__ = mock.Mock(return_value=mock_batch)
    mock_client.batch.return_value.__exit__ = mock.Mock(return_value=False)

    source_records = []

    result = sync_collection(mock_client, source_records)

    assert result == 0
    mock_batch.delete_record.assert_called_once_with(id="test-1")


@mock.patch("ai_window_prompts_updater.ENVIRONMENT", "dev")
def test_sync_collection_dev_auto_approve(capsys):
    mock_client = mock.Mock()
    mock_client.get_records.return_value = []
    mock_batch = mock.Mock()
    mock_batch.results.return_value = [{}]
    mock_client.batch.return_value.__enter__ = mock.Mock(return_value=mock_batch)
    mock_client.batch.return_value.__exit__ = mock.Mock(return_value=False)

    source_records = [{"id": "test-1", "data": "foo"}]

    result = sync_collection(mock_client, source_records)

    assert result == 0
    mock_client.request_review.assert_called_once()
    mock_client.approve_changes.assert_called_once()
    output = capsys.readouterr().out
    assert "🟢 Self-approving changes on dev" in output


def test_sync_collection_fetch_error(capsys):
    mock_client = mock.Mock()
    mock_client.get_records.side_effect = KintoException("Server error")

    result = sync_collection(mock_client, [])

    assert result == 1
    output = capsys.readouterr().out
    assert "Failed to fetch existing records" in output


def test_sync_collection_batch_error(capsys):
    mock_client = mock.Mock()
    mock_client.get_records.return_value = []
    mock_batch_context = mock.MagicMock()
    mock_batch_context.__enter__.side_effect = KintoException("Batch failed")
    mock_client.batch.return_value = mock_batch_context

    source_records = [{"id": "test-1", "data": "foo"}]

    result = sync_collection(mock_client, source_records)

    assert result == 1
    output = capsys.readouterr().out
    assert "Failed to apply changes" in output


def test_sync_collection_review_error(capsys):
    mock_client = mock.Mock()
    mock_client.get_records.return_value = []
    mock_batch = mock.Mock()
    mock_batch.results.return_value = [{}]
    mock_client.batch.return_value.__enter__ = mock.Mock(return_value=mock_batch)
    mock_client.batch.return_value.__exit__ = mock.Mock(return_value=False)
    mock_client.request_review.side_effect = KintoException("Review failed")

    source_records = [{"id": "test-1", "data": "foo"}]

    result = sync_collection(mock_client, source_records)

    assert result == 1
    output = capsys.readouterr().out
    assert "Failed to update collection status" in output


# Integration tests for main function
@mock.patch("ai_window_prompts_updater.sync_collection")
@mock.patch("ai_window_prompts_updater.fetch_current_prompts")
@mock.patch("ai_window_prompts_updater.clone_repo")
@mock.patch("ai_window_prompts_updater.Client")
def test_main_full_success(mock_client_class, mock_clone, mock_fetch, mock_sync):
    mock_client = mock.Mock()
    mock_client.server_info.return_value = {"user": {"id": "test@example.com"}}

    mock_client_class.return_value = mock_client
    mock_clone.return_value = Path("/tmp/test"), "prod"
    mock_fetch.return_value = [{"id": "test-1"}]
    mock_sync.return_value = 0

    result = main()

    assert result == 0
    mock_clone.assert_called_once()
    mock_fetch.assert_called_once()
    mock_sync.assert_called_once()


@mock.patch("ai_window_prompts_updater.clone_repo")
@mock.patch("ai_window_prompts_updater.Client")
def test_main_clone_failure(mock_client_class, mock_clone):
    mock_client = mock.Mock()
    mock_client.server_info.return_value = {"user": {"id": "test@example.com"}}
    mock_client_class.return_value = mock_client
    mock_clone.return_value = None, None

    result = main()

    assert result == 1


@mock.patch("ai_window_prompts_updater.sync_collection")
@mock.patch("ai_window_prompts_updater.fetch_current_prompts")
@mock.patch("ai_window_prompts_updater.clone_repo")
@mock.patch("ai_window_prompts_updater.Client")
def test_main_sync_failure(mock_client_class, mock_clone, mock_fetch, mock_sync):
    mock_client = mock.Mock()
    mock_client.server_info.return_value = {"user": {"id": "test@example.com"}}
    mock_client_class.return_value = mock_client
    mock_clone.return_value = Path("/tmp/test"), "prod"
    mock_fetch.return_value = [{"id": "test-1"}]
    mock_sync.return_value = 1

    result = main()

    assert result == 1


@mock.patch("ai_window_prompts_updater.Client")
def test_main_connection_failure(mock_client_class, capsys):
    mock_client = mock.Mock()
    mock_client.server_info.side_effect = Exception("Connection refused")
    mock_client_class.return_value = mock_client

    result = main()

    assert result == 1
    output = capsys.readouterr().out
    assert "Failed to connect to Remote Settings server" in output
