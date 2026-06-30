import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from kinto_http import KintoException

from ai_window_prompts_updater import (
    clone_repo,
    collect_prompts_and_params,
    collect_v2_records,
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
    assert "Anonymous" in capsys.readouterr().out


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
    assert "Repository cloned successfully" in capsys.readouterr().out
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


@mock.patch("ai_window_prompts_updater.GIT_TOKEN", None)
@mock.patch("ai_window_prompts_updater.PROMPTS_REPO", "https://github.com/test/repo.git")
@mock.patch("ai_window_prompts_updater.subprocess.run")
def test_clone_repo_without_token(mock_run):
    mock_run.return_value = mock.Mock(returncode=0, stderr="")

    clone_repo("prod")

    # No token: clone the public repo anonymously (plain URL, no credentials).
    call_args = mock_run.call_args[0][0]
    assert "https://github.com/test/repo.git" in call_args
    assert "@" not in " ".join(call_args)


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
@pytest.fixture
def temp_v2_prompts_dir():
    """Create a temporary repo with both prompts/ (legacy) and prompts_v2/ trees."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "repo"

        legacy = repo_path / "prompts" / "chat" / "v1"
        legacy.mkdir(parents=True)
        with open(legacy / "claude.3.5.json", "w") as f:
            json.dump(
                {"feature": "chat", "model": "claude.3.5", "parameters": {"temperature": 0.7}},
                f,
            )
        with open(legacy / "claude.3.5.md", "w") as f:
            f.write("Legacy prompt")

        v2 = repo_path / "prompts_v2"
        identity = v2 / "features" / "chat" / "identity" / "v1"
        identity.mkdir(parents=True)
        with open(identity / "generic.json", "w") as f:
            json.dump({"version": "1.0"}, f)
        with open(identity / "generic.md", "w") as f:
            f.write("# Identity\nYou are Smart Window.")

        model_details = v2 / "features" / "chat" / "model-details" / "v1"
        model_details.mkdir(parents=True)
        with open(model_details / "qwen3-235b-a22b-instruct-2507-maas.json", "w") as f:
            json.dump({"version": "1.0"}, f)
        with open(model_details / "qwen3-235b-a22b-instruct-2507-maas.md", "w") as f:
            f.write("Qwen-specific cutoff.")

        params = v2 / "features" / "chat" / "params" / "v1"
        params.mkdir(parents=True)
        with open(params / "generic.json", "w") as f:
            json.dump(
                {
                    "version": "1.0",
                    "temperature": 1.0,
                    "purpose": "chat",
                    "service_type": "ai",
                },
                f,
            )

        tab = v2 / "features" / "browser-context" / "tab" / "v1"
        tab.mkdir(parents=True)
        with open(tab / "generic.json", "w") as f:
            json.dump({"version": "1.0"}, f)
        with open(tab / "generic.md", "w") as f:
            f.write("This is my active tab")

        kit = v2 / "skills" / "kit" / "v1"
        kit.mkdir(parents=True)
        with open(kit / "generic.json", "w") as f:
            json.dump({"version": "1.0", "description": "Mascot info"}, f)
        with open(kit / "generic.md", "w") as f:
            f.write("Kit is a Firefox mascot")

        yield repo_path


def test_collect_v2_records_module(temp_v2_prompts_dir):
    records = collect_v2_records(temp_v2_prompts_dir / "prompts_v2")
    identity = next(r for r in records if r.get("module") == "identity")
    assert identity["id"] == "chat--identity--generic--v1"
    assert identity["kind"] == "module"
    assert identity["feature"] == "chat"
    assert identity["model"] == "generic"
    assert identity["version"] == "1.0"  # no manifest in fixture -> dir-derived
    assert identity["prompts"] == "# Identity\nYou are Smart Window."


def test_collect_v2_records_module_version_from_manifest(temp_v2_prompts_dir):
    # The params manifest is the source of truth for module versions: a module
    # record's version is stamped from the manifest entry (matched by feature +
    # module + major), not from its directory name. The id stays keyed by the
    # dir (one record per major), so only the version field reflects the minor.
    params_dir = temp_v2_prompts_dir / "prompts_v2" / "features" / "chat" / "params" / "v1"
    with open(params_dir / "generic.json", "w") as f:
        json.dump(
            {
                "version": "1.0",
                "temperature": 1.0,
                "modules": [
                    {"name": "identity", "version": "1.4"},
                    {"name": "model-details", "version": "1.0"},
                ],
            },
            f,
        )
    records = collect_v2_records(temp_v2_prompts_dir / "prompts_v2")
    by_module = {r.get("module"): r for r in records if r.get("kind") == "module"}
    assert by_module["identity"]["version"] == "1.4"
    assert by_module["identity"]["id"] == "chat--identity--generic--v1"
    # A module the manifest does not name keeps its dir-derived version.
    assert by_module["model-details"]["version"] == "1.0"
    # The chat manifest must not bleed into another feature's modules.
    tab = next(r for r in records if r.get("feature") == "browser-context")
    assert tab["version"] == "1.0"


def test_collect_v2_records_rejects_bad_manifest_version(temp_v2_prompts_dir):
    # Manifest versions must be "major.minor"; a bare integer is rejected so the
    # stamped record version always parses the same way Firefox selects on.
    params_dir = temp_v2_prompts_dir / "prompts_v2" / "features" / "chat" / "params" / "v1"
    with open(params_dir / "generic.json", "w") as f:
        json.dump({"version": "1.0", "modules": [{"name": "identity", "version": "1"}]}, f)

    with pytest.raises(ValueError, match="major.minor"):
        collect_v2_records(temp_v2_prompts_dir / "prompts_v2")


def test_collect_v2_records_rejects_duplicate_manifest_module(temp_v2_prompts_dir):
    params_dir = temp_v2_prompts_dir / "prompts_v2" / "features" / "chat" / "params" / "v1"
    with open(params_dir / "generic.json", "w") as f:
        json.dump(
            {
                "version": "1.0",
                "modules": [
                    {"name": "identity", "version": "1.0"},
                    {"name": "identity", "version": "1.1"},
                ],
            },
            f,
        )

    with pytest.raises(ValueError, match="more than once"):
        collect_v2_records(temp_v2_prompts_dir / "prompts_v2")


def test_collect_v2_records_rejects_manifest_module_without_content_dir(
    temp_v2_prompts_dir,
):
    # Manifest names identity at major 2, but only an identity/v1 dir exists, so
    # the manifest version can't be honored -> reject at publish time rather than
    # ship a record that hard-fails Firefox assembly.
    params_dir = temp_v2_prompts_dir / "prompts_v2" / "features" / "chat" / "params" / "v1"
    with open(params_dir / "generic.json", "w") as f:
        json.dump(
            {"version": "1.0", "modules": [{"name": "identity", "version": "2.0"}]}, f
        )

    with pytest.raises(ValueError, match="no matching content director"):
        collect_v2_records(temp_v2_prompts_dir / "prompts_v2")


def test_module_version_is_major_only():
    from ai_window_prompts_updater import _module_version

    assert _module_version("v1") == "1.0"
    assert _module_version("v2") == "2.0"
    # A dotted dir name must not corrupt into "25.0" (the module minor lives in
    # the params manifest, not the directory).
    assert _module_version("v2.5") == "2.0"


def test_collect_v2_records_model_specific(temp_v2_prompts_dir):
    records = collect_v2_records(temp_v2_prompts_dir / "prompts_v2")
    qwen = next(r for r in records if r.get("module") == "model-details")
    assert qwen["id"] == "chat--model-details--qwen3-235b-a22b-instruct-2507-maas--v1"
    assert qwen["model"] == "qwen3-235b-a22b-instruct-2507-maas"


def test_collect_v2_records_browser_context(temp_v2_prompts_dir):
    records = collect_v2_records(temp_v2_prompts_dir / "prompts_v2")
    tab = next(r for r in records if r.get("feature") == "browser-context")
    assert tab["id"] == "browser-context--tab--generic--v1"
    assert tab["kind"] == "module"
    assert tab["module"] == "tab"


def test_collect_v2_records_skill(temp_v2_prompts_dir):
    records = collect_v2_records(temp_v2_prompts_dir / "prompts_v2")
    kit = next(r for r in records if r.get("kind") == "skill")
    assert kit["id"] == "skill--kit--generic--v1"
    assert kit["name"] == "kit"
    assert kit["version"] == "1.0"
    assert kit["description"] == "Mascot info"
    assert kit["prompts"] == "Kit is a Firefox mascot"


def test_collect_v2_records_params(temp_v2_prompts_dir):
    records = collect_v2_records(temp_v2_prompts_dir / "prompts_v2")
    params = next(r for r in records if r.get("kind") == "params")
    assert params["id"] == "chat--params--generic--v1"
    assert params["feature"] == "chat"
    assert params["model"] == "generic"
    assert params["temperature"] == 1.0
    assert params["purpose"] == "chat"
    assert params["service_type"] == "ai"
    assert "prompt" not in params


def test_collect_v2_records_params_loads_all_keys(temp_v2_prompts_dir):
    # Drop in a JSON with arbitrary keys; all should be preserved on the record.
    params_dir = temp_v2_prompts_dir / "prompts_v2" / "features" / "chat" / "params" / "v1"
    with open(params_dir / "generic.json", "w") as f:
        json.dump(
            {
                "version": "1.0",
                "temperature": 0.7,
                "top_p": 0.95,
                "custom_flag": True,
                "nested": {"a": 1},
            },
            f,
        )
    records = collect_v2_records(temp_v2_prompts_dir / "prompts_v2")
    params = next(r for r in records if r.get("kind") == "params")
    assert params["temperature"] == 0.7
    assert params["top_p"] == 0.95
    assert params["custom_flag"] is True
    assert params["nested"] == {"a": 1}


def test_collect_v2_records_params_per_model(temp_v2_prompts_dir):
    # A params dir can have one JSON per model alongside generic.json.
    params_dir = temp_v2_prompts_dir / "prompts_v2" / "features" / "chat" / "params" / "v1"
    with open(params_dir / "qwen3-235b-a22b-instruct-2507-maas.json", "w") as f:
        json.dump({"version": "1.0", "temperature": 0.5}, f)

    records = collect_v2_records(temp_v2_prompts_dir / "prompts_v2")
    params_records = [r for r in records if r.get("kind") == "params"]
    by_model = {r["model"]: r for r in params_records}
    assert "generic" in by_model
    assert "qwen3-235b-a22b-instruct-2507-maas" in by_model
    assert by_model["generic"]["id"] == "chat--params--generic--v1"
    assert (
        by_model["qwen3-235b-a22b-instruct-2507-maas"]["id"]
        == "chat--params--qwen3-235b-a22b-instruct-2507-maas--v1"
    )
    assert by_model["qwen3-235b-a22b-instruct-2507-maas"]["temperature"] == 0.5


def test_collect_v2_records_params_in_other_feature(temp_v2_prompts_dir):
    # params can appear under any feature, not just chat.
    bc_params = (
        temp_v2_prompts_dir / "prompts_v2" / "features" / "browser-context" / "params" / "v1"
    )
    bc_params.mkdir(parents=True)
    with open(bc_params / "generic.json", "w") as f:
        json.dump({"version": "1.0", "max_tokens": 200}, f)

    records = collect_v2_records(temp_v2_prompts_dir / "prompts_v2")
    bc = next(
        r for r in records if r.get("kind") == "params" and r.get("feature") == "browser-context"
    )
    assert bc["id"] == "browser-context--params--generic--v1"
    assert bc["max_tokens"] == 200


def test_collect_v2_records_params_rejects_reserved_keys(temp_v2_prompts_dir):
    # Reserved keys (id/kind/feature/model) must not silently clobber the
    # computed identity fields; the updater should refuse to ingest such a file.
    params_dir = temp_v2_prompts_dir / "prompts_v2" / "features" / "chat" / "params" / "v1"
    with open(params_dir / "generic.json", "w") as f:
        json.dump({"version": "1.0", "feature": "rogue", "temperature": 0.1}, f)

    with pytest.raises(ValueError, match="reserved key"):
        collect_v2_records(temp_v2_prompts_dir / "prompts_v2")


def test_collect_v2_records_normalizes_dots_in_model_name(temp_v2_prompts_dir):
    gemini = temp_v2_prompts_dir / "prompts_v2" / "features" / "chat" / "model-details" / "v1"
    with open(gemini / "gemini-2.5-flash-lite.json", "w") as f:
        json.dump({"version": "1.0"}, f)
    with open(gemini / "gemini-2.5-flash-lite.md", "w") as f:
        f.write("Gemini cutoff")

    records = collect_v2_records(temp_v2_prompts_dir / "prompts_v2")
    rec = next(r for r in records if r.get("model") == "gemini-2.5-flash-lite")
    assert rec["id"] == "chat--model-details--gemini-2-5-flash-lite--v1"


def test_collect_v2_records_no_v2_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_repo = Path(temp_dir)
        records = collect_v2_records(empty_repo / "prompts_v2")
        assert records == []


def test_fetch_current_prompts_includes_v2(temp_v2_prompts_dir, capsys):
    records = fetch_current_prompts(temp_v2_prompts_dir)
    legacy = [r for r in records if r.get("id", "").startswith("chat--") and "kind" not in r]
    v2 = [r for r in records if r.get("kind") in {"module", "skill", "params"}]
    assert len(legacy) == 1
    assert len(v2) >= 4
    output = capsys.readouterr().out
    assert "v2 prompt records" in output


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
    assert "Found 2 prompt records" in output
    assert "Cleaned up temporary directory" in output


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
    assert "Self-approving changes on dev" in output


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
