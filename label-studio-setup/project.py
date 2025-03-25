import argparse
import os
from pathlib import Path
from label_studio_sdk import LabelStudio
from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIEW_FILE = os.path.join(SCRIPT_DIR, "view.yaml")
LOCAL_STORAGE_PATH = Path(os.getenv("DATASET_PATH")).resolve()

parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="Name of the project to create")
parser.add_argument(
    "--no-ml", action="store_true", help="Do not connect the ml backend"
)
parser.add_argument(
    "--no-local", action="store_true", help="Do not connect the local storage"
)
parser.add_argument(
    "--sync",
    action="store_true",
    help="This will only sync the local storage and not create a project",
)
args = parser.parse_args()

PROJECT_NAME = args.name

if not LABEL_STUDIO_URL or not API_KEY:
    print("Error: Missing LABEL_STUDIO_URL or API_KEY in .env")
    exit(1)

if not os.path.exists(VIEW_FILE):
    print(f"Error: {VIEW_FILE} not found in the current directory.")
    exit(1)

with open(VIEW_FILE, "r", encoding="utf-8") as file:
    label_config = file.read()

ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

try:
    if args.sync:
        projects = ls.projects.list(title=PROJECT_NAME)
        project = None
        for item in projects:
            if item.title == PROJECT_NAME:
                project = item

        if not project:
            print(f"Error: Project '{PROJECT_NAME}' not found.")
            exit(1)

        print(f"Synchronizing project '{project.title}':")

        storages = localStorage = ls.import_storage.local.list(project=project.id)

        count = 0
        for storage in storages:
            print(f"Syncing storage '{storage.title}'...")
            sync = ls.import_storage.local.sync(storage.id)
            count += sync.last_sync_count
        print(f"Synced storages successfully. Synced {count} items.")
        exit(0)

    project = ls.projects.create(
        title=PROJECT_NAME,
        description="Automated darts scoring",
        label_config=label_config,
    )
    print(
        f"Project '{project.title}' created successfully with labeling interface from '{VIEW_FILE}'."
    )

    if args.no_ml:
        print("Skipping ML backend connection.")
    else:
        print("Connecting to ML backend...")
        mlBackend = ls.ml.create(
            title="Yolo Dart Keypoint",
            description="Yolo backend for dart keypoint detection ",
            project=project.id,
            url="http://yolo-ml-backend:9090",
        )
        print("Successfully connected to ML backend.")

    if args.no_local:
        print("Skipping local storage connection.")
    else:
        print("Connecting to local storage...")
        localStorage = ls.import_storage.local.create(
            title="Local Storage",
            description=f"Local storage at {str(LOCAL_STORAGE_PATH)}",
            path=f"/home/user/dataset",
            regex_filter=".*(jpe?g|png|tiff|webp)",
            use_blob_urls=True,
            project=project.id,
        )
        print(f"Successfully connected to local storage at {str(LOCAL_STORAGE_PATH)}.")
        print("Syncing local storage...")
        sync = ls.import_storage.local.sync(localStorage.id)

        if sync.status == "failed":
            print(f"Error: Failed to sync local storage.")
            print(sync.traceback)
            exit(1)

        print(
            f"Local storage synced successfully. Synced {sync.last_sync_count} items."
        )

except Exception as e:
    print(f"Error: Failed to setup project '{PROJECT_NAME}'.")
    print(e)
    exit(1)
