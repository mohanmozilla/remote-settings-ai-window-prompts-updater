import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import sentry_sdk
from kinto_http import Client, KintoException
from kinto_http.utils import collection_diff
from sentry_sdk.integrations.gcp import GcpIntegration


# Required environment variables
AUTHORIZATION = os.getenv("AUTHORIZATION", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "local").lower()
SERVER = os.getenv(
    "SERVER",
    {
        "local": "http://localhost:8888/v1",
        "dev": "https://remote-settings-dev.allizom.org/v1",
        "stage": "https://remote-settings.allizom.org/v1",
        "prod": "https://remote-settings.mozilla.org/v1",
    }[ENVIRONMENT],
)
IS_DRY_RUN = os.getenv("DRY_RUN", "0") in "1yY"
SENTRY_DSN = os.getenv("SENTRY_DSN")
SENTRY_ENV = os.getenv("SENTRY_ENV", ENVIRONMENT)
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))

if ENVIRONMENT not in {"local", "dev", "stage", "prod"}:
    raise ValueError(f"'ENVIRONMENT={ENVIRONMENT}' is not a valid value")

# Add token support to line 64
GIT_TOKEN = os.getenv("GIT_TOKEN")

# Constants for collection
BUCKET = "main-workspace"
AI_WINDOW_PROMPTS_COLLECTION = "ai-window-prompts"
PROMPTS_REPO = "https://github.com/Firefox-AI/ai-window-remote-settings-prompts.git"


def clone_repo(branch):
    """
    Clone the prompts repo and fetch prompts from it.

    Returns a list of records suitable for Remote Settings, where each record
    contains the metadata from the JSON file and the prompt content from the MD file.
    """

    # Create a temporary directory for cloning
    temp_dir = tempfile.mkdtemp(prefix="prompts_")
    repo_path = Path(temp_dir) / "ai-window-remote-settings-prompts"

    try:
        branch_to_clone = branch if branch == "stage" else "prod"
        print(f"Cloning {PROMPTS_REPO}...")
        print(f"Branch: {branch_to_clone}")

        if GIT_TOKEN:
            git_url = PROMPTS_REPO.replace("https://", f"https://{GIT_TOKEN}@")
        else:
            # Public repo: clone anonymously when no token is available (e.g. fork-PR CI).
            git_url = PROMPTS_REPO

        # Clone stage to stage, prod gets cloned to dev and prod
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch_to_clone, git_url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone repository: {result.stderr}")

        print("Repository cloned successfully")

        return repo_path, branch_to_clone
    except Exception as e:
        print(f"ERROR cloning repo: {e}")
        return "", ""


def fetch_current_prompts(repo_path):
    prompts_dir = repo_path / "prompts"
    prompts_v2_dir = repo_path / "prompts_v2"

    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found at {prompts_dir}")
    try:
        records = collect_prompts_and_params(prompts_dir)
        if prompts_v2_dir.exists():
            v2_records = collect_v2_records(prompts_v2_dir)
            print(f"Found {len(v2_records)} v2 prompt records")
            records.extend(v2_records)
    finally:
        if repo_path.parent.exists():
            shutil.rmtree(repo_path.parent)
            print("Cleaned up temporary directory")

        print(f"Found {len(records)} prompt records")
    return records


def get_item(major_version_dir, model_name):
    with open(major_version_dir / f"{model_name}.json", "r") as f:
        data = json.load(f)
    with open(major_version_dir / f"{model_name}.md", "r") as f:
        prompt = f.read()

    # Add the prompt content
    data["prompts"] = prompt

    # Create a unique ID based on feature, version, and model
    data["id"] = f"{data['feature']}--{data['model'].replace('.', '-')}--{major_version_dir.stem}"
    data["parameters"] = json.dumps(data["parameters"])

    return data


def collect_prompts_and_params(prompts_dir):
    items = []
    for feature_dir in prompts_dir.iterdir():
        for major_version_dir in feature_dir.iterdir():
            for f in major_version_dir.iterdir():
                if f.suffix == ".md":
                    continue
                items.append(get_item(major_version_dir, f.stem))
    return items


# ---------------------------------------------------------------------------
# prompts_v2 ingestion
#
# Layout:
#   features/<feature>/<module>/v#/<model|generic>.{json,md}
#     - regular modules: .md is the prompt body; optional .json sidecar (one
#       sidecar per model, paired by stem with the .md file)
#     - module == "params": JSON-only generation params (one per model). The
#       params module may live under any feature, not only "chat".
#   skills/<name>/v#/<model|generic>.{json,md}
#
# Records carry a `kind` discriminator: "module", "params", or "skill". The
# enum is open-ended — additional kinds will be added as more legacy prompts
# migrate into prompts_v2 (e.g. memories-relevant-context, title-generation).
# Features are open-ended too — new features drop in without an updater change.


def _normalize_model(stem):
    return stem.replace(".", "-")


def _module_version(version_segment):
    # "{major}.0" from a "v1" dir name. A module's minor is carried by the
    # params manifest, not its directory, so the dir-derived fallback is
    # major-only (and a dotted dir name can't corrupt the value).
    major = _major_of(version_segment)
    return f"{major}.0" if major is not None else "1.0"


def _major_of(version_segment):
    # Major version (int) from "v1", "1.0", "8.1", etc.; None when unparseable.
    head = str(version_segment).lstrip("vV").split(".")[0]
    return int(head) if head.isdigit() else None


def _read_json_if_exists(path):
    """Return the parsed JSON sidecar at ``path`` or an empty dict when the
    file is missing or empty. Returning {} (instead of None) lets call sites
    use ``json_data.get("version", "1.0")`` uniformly without ``or {}`` guards.
    """
    if path is None or not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f) or {}


def _pair_files_by_stem(directory):
    pairs = {}
    for f in directory.iterdir():
        if f.is_file() and f.suffix in (".json", ".md"):
            pairs.setdefault(f.stem, {})[f.suffix] = f
    return pairs


def _collect_manifest_versions(prompts_v2_dir):
    """Map (feature, module, major) -> full version string, read from the
    ``modules`` manifest in each feature's params JSON. The params manifest is
    the single source of truth for module versions (Firefox selects each module
    by the version the manifest names), so module records are stamped from here
    rather than from their own directory name. Last write wins when more than
    one params file names the same (feature, module, major)."""
    versions = {}
    features_dir = prompts_v2_dir / "features"
    if not features_dir.exists():
        return versions
    for feature_dir in sorted(features_dir.iterdir()):
        params_root = feature_dir / "params"
        if not params_root.is_dir():
            continue
        for version_dir in sorted(params_root.iterdir()):
            if not version_dir.is_dir():
                continue
            for f in sorted(version_dir.glob("*.json")):
                seen = set()
                for entry in _read_json_if_exists(f).get("modules", []):
                    name, ver = entry.get("name"), entry.get("version")
                    if not name or ver is None:
                        raise ValueError(
                            f"{f}: each `modules` entry needs a 'name' and 'version'"
                        )
                    if not re.fullmatch(r"v?\d+\.\d+", str(ver)):
                        raise ValueError(
                            f"{f}: module '{name}' version '{ver}' must be "
                            "'major.minor' (e.g. '1.0')"
                        )
                    if name in seen:
                        raise ValueError(
                            f"{f}: module '{name}' is listed more than once in `modules`"
                        )
                    seen.add(name)
                    versions[(feature_dir.name, name, _major_of(ver))] = str(ver)
    return versions


def _require_manifest_content(manifest_versions, available_modules):
    # Every (feature, module, major) a params manifest names must have a
    # matching content directory; otherwise the manifest version can't be
    # honored and Firefox would fail to assemble the prompt. Fail here (publish
    # time) rather than shipping a record that hard-fails at runtime.
    missing = sorted(k for k in manifest_versions if k not in available_modules)
    if missing:
        details = ", ".join(f"{feat}/{mod} (major {maj})" for feat, mod, maj in missing)
        raise ValueError(
            f"params manifest names modules with no matching content directory: "
            f"{details}. Add the module's v<major> directory (with a .md), or "
            "correct the manifest version."
        )


def collect_v2_records(prompts_v2_dir):
    items = []
    manifest_versions = _collect_manifest_versions(prompts_v2_dir)
    available_modules = set()
    features_dir = prompts_v2_dir / "features"
    if features_dir.exists():
        for feature_dir in sorted(features_dir.iterdir()):
            if not feature_dir.is_dir():
                continue
            for module_dir in sorted(feature_dir.iterdir()):
                if not module_dir.is_dir():
                    continue
                for version_dir in sorted(module_dir.iterdir()):
                    if not version_dir.is_dir():
                        continue
                    records = _collect_v2_module_records(
                        version_dir,
                        feature_dir.name,
                        module_dir.name,
                        version_dir.name,
                        manifest_versions,
                    )
                    items.extend(records)
                    major = _major_of(version_dir.name)
                    if module_dir.name != "params" and records and major is not None:
                        available_modules.add(
                            (feature_dir.name, module_dir.name, major)
                        )

    _require_manifest_content(manifest_versions, available_modules)

    skills_dir = prompts_v2_dir / "skills"
    if skills_dir.exists():
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            for version_dir in sorted(skill_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                items.extend(
                    _collect_v2_skill_records(version_dir, skill_dir.name, version_dir.name)
                )

    return items


def _collect_v2_module_records(version_dir, feature, module, version, manifest_versions):
    if module == "params":
        return _collect_v2_params_records(version_dir, feature, version)

    # Version is stamped from the params manifest for this (feature, module,
    # major); fall back to the dir-derived version when no manifest names it.
    record_version = manifest_versions.get(
        (feature, module, _major_of(version)), _module_version(version)
    )

    items = []
    for stem, paths in sorted(_pair_files_by_stem(version_dir).items()):
        md_path = paths.get(".md")
        if md_path is None:
            continue  # no prompt content; skip
        items.append(
            {
                "id": f"{feature}--{module}--{_normalize_model(stem)}--{version}",
                "kind": "module",
                "feature": feature,
                "module": module,
                "model": stem,
                "version": record_version,
                "prompts": md_path.read_text(),
            }
        )
    return items


PARAMS_RESERVED_KEYS = frozenset({"id", "kind", "feature", "model"})


def _collect_v2_params_records(version_dir, feature, version):
    items = []
    for f in sorted(version_dir.iterdir()):
        if not f.is_file() or f.suffix != ".json":
            continue
        json_data = _read_json_if_exists(f)
        if not json_data:
            continue
        stem = f.stem
        conflicts = PARAMS_RESERVED_KEYS & set(json_data)
        if conflicts:
            raise ValueError(
                f"{f}: params JSON contains reserved key(s) {sorted(conflicts)} "
                "which would clobber the record's computed identity fields; "
                "remove them from the params file."
            )
        items.append(
            {
                **json_data,
                "id": f"{feature}--params--{_normalize_model(stem)}--{version}",
                "kind": "params",
                "feature": feature,
                "model": stem,
            }
        )
    return items


def _collect_v2_skill_records(version_dir, name, version):
    items = []
    for stem, paths in sorted(_pair_files_by_stem(version_dir).items()):
        md_path = paths.get(".md")
        if md_path is None:
            continue
        json_data = _read_json_if_exists(paths.get(".json"))
        items.append(
            {
                "id": f"skill--{name}--{_normalize_model(stem)}--{version}",
                "kind": "skill",
                "name": name,
                "model": stem,
                "version": _module_version(version),
                "description": json_data.get("description", ""),
                "prompts": md_path.read_text(),
            }
        )
    return items


def sync_collection(client, source_records):
    print("Fetching current destination records...")
    try:
        dest_records = client.get_records()
    except KintoException as e:
        print(f"Failed to fetch existing records: {e}")
        return 1

    # Compute the diff
    to_create, to_update, to_delete = collection_diff(source_records, dest_records)

    has_changes = to_create or to_update or to_delete
    if not has_changes:
        print("Records are already in sync. Nothing to do.")
        return 0

    print(
        f"Applying {len(to_create)} creates, {len(to_update)} updates, {len(to_delete)} deletes..."
    )
    try:
        with client.batch() as batch:
            for record in to_create:
                batch.create_record(data=record)
            for _, record in to_update:
                record.pop("last_modified", None)
                batch.update_record(data=record)
            for record in to_delete:
                batch.delete_record(id=record["id"])
        ops_count = len(batch.results())
        print(f"Batch {ops_count} operations applied.")
    except KintoException as e:
        print(f"Failed to apply changes: {e}")
        return 1

    try:
        if ENVIRONMENT == "dev":
            print("Self-approving changes on dev...")
            client.request_review(message="r?")
            client.approve_changes()
            print("Changes self-approved.")
        else:
            print("Requesting review...")
            client.request_review(message="r?")
            print("Review requested.")
    except KintoException as e:
        print(f"Failed to update collection status: {e}")
        return 1

    return 0


def main():
    if SENTRY_DSN:
        # Initialize Sentry for error reporting.
        sentry_sdk.init(SENTRY_DSN, integrations=[GcpIntegration()], environment=SENTRY_ENV)
    else:
        print("Sentry is not configured. Set SENTRY_DSN environment variable to enable it.")

    remote_settings_client = Client(
        server_url=SERVER,
        auth=AUTHORIZATION,
        bucket=BUCKET,
        collection=AI_WINDOW_PROMPTS_COLLECTION,
        dry_mode=IS_DRY_RUN,
    )
    try:
        print("Checking credentials...", end="")
        server_info = remote_settings_client.server_info()
        print("")
        if "user" in server_info:
            print(f"Logged in as {server_info['user']['id']}")
        else:
            print("Anonymous access")
    except Exception as e:
        print(f"Failed to connect to Remote Settings server: {e}")
        return 1

    print("\n=== Processing prompts ===")
    print("Fetching prompts ...")
    repo_path, _ = clone_repo(ENVIRONMENT)
    if not repo_path:
        return 1
    prompts = fetch_current_prompts(repo_path)

    result = sync_collection(remote_settings_client, prompts)
    if result != 0:
        return result

    return 0


if __name__ == "__main__":
    sys.exit(main())
