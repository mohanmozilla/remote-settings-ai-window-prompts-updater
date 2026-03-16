import json
import os
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


def clone_repo(git_url):
    """
    Clone the prompts repo and fetch prompts from it.

    Returns a list of records suitable for Remote Settings, where each record
    contains the metadata from the JSON file and the prompt content from the MD file.
    """

    # Create a temporary directory for cloning
    temp_dir = tempfile.mkdtemp(prefix="prompts_")
    repo_path = Path(temp_dir) / "ai-window-remote-settings-prompts"

    try:
        print(f"🔄 Cloning {git_url}...")

        if GIT_TOKEN:
            git_url = git_url.replace("https://", f"https://{GIT_TOKEN}@")
        else:
            raise RuntimeError("GIT_TOKEN not found")

        # Clone the repository
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "main", git_url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone repository: {result.stderr}")

        print("✅ Repository cloned successfully")

        return repo_path
    except Exception as e:
        print(f"ERROR cloning repo: {e}")
        return ""


def fetch_current_prompts(repo_path):
    prompts_dir = repo_path / "prompts"

    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found at {prompts_dir}")
    try:
        records = collect_prompts_and_params(prompts_dir)
    finally:
        if repo_path.parent.exists():
            shutil.rmtree(repo_path.parent)
            print("🧹 Cleaned up temporary directory")

        print(f"📦 Found {len(records)} prompt records")
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


def sync_collection(client, source_records):
    print("📥 Fetching current destination records...")
    try:
        dest_records = client.get_records()
    except KintoException as e:
        print(f"❌ Failed to fetch existing records: {e}")
        return 1

    # Compute the diff
    to_create, to_update, to_delete = collection_diff(source_records, dest_records)

    has_changes = to_create or to_update or to_delete
    if not has_changes:
        print("✅ Records are already in sync. Nothing to do.")
        return 0

    print(
        f"🔧 Applying {len(to_create)} creates, {len(to_update)} updates, {len(to_delete)} deletes..."
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
        print(f"✅ Batch {ops_count} operations applied.")
    except KintoException as e:
        print(f"❌ Failed to apply changes: {e}")
        return 1

    try:
        if ENVIRONMENT == "dev":
            print("🟢 Self-approving changes on dev...")
            client.request_review(message="r?")
            client.approve_changes()
            print("✅ Changes self-approved.")
        else:
            print("📤 Requesting review...")
            client.request_review(message="r?")
            print("✅ Review requested.")
    except KintoException as e:
        print(f"❌ Failed to update collection status: {e}")
        return 1

    return 0


def main():
    if SENTRY_DSN:
        # Initialize Sentry for error reporting.
        sentry_sdk.init(SENTRY_DSN, integrations=[GcpIntegration()], environment=SENTRY_ENV)
    else:
        print("⚠️ Sentry is not configured. Set SENTRY_DSN environment variable to enable it.")

    remote_settings_client = Client(
        server_url=SERVER,
        auth=AUTHORIZATION,
        bucket=BUCKET,
        collection=AI_WINDOW_PROMPTS_COLLECTION,
        dry_mode=IS_DRY_RUN,
    )
    try:
        print("🔐 Checking credentials...", end="")
        server_info = remote_settings_client.server_info()
        print("✅")
        if "user" in server_info:
            print(f"👤 Logged in as {server_info['user']['id']}")
        else:
            print("⚠️ Anonymous access")
    except Exception as e:
        print(f"❌ Failed to connect to Remote Settings server: {e}")
        return 1

    print("\n=== Processing prompts ===")
    print("📥 Fetching prompts ...")
    repo_path = clone_repo(PROMPTS_REPO)
    if not repo_path:
        return 1
    prompts = fetch_current_prompts(repo_path)

    result = sync_collection(remote_settings_client, prompts)
    if result != 0:
        return result

    return 0


if __name__ == "__main__":
    sys.exit(main())
