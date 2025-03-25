import os
import httpx
import argparse
from label_studio_sdk import LabelStudio
from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
ML_BACKEND_URL = "http://yolo-ml-backend:9090"
ML_BACKEND_TITLE = "YOLO ML Backend"

if not LABEL_STUDIO_URL or not API_KEY:
    print("Error: Missing LABEL_STUDIO_URL or API_KEY in .env")
    exit(1)

parser = argparse.ArgumentParser()
parser.add_argument('project', type=str, required=True, help='Name of the project')
args = parser.parse_args()

PROJECT_NAME = args.project

ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

try:
    projects = ls.projects.list(title=PROJECT_NAME)
    project = None
    for item in projects:
        if item.title == PROJECT_NAME:
            project = item

    if not project:
        print(f"Error: Project '{PROJECT_NAME}' not found.")
        exit(1)

    print("Connecting to ML backend...")
    mlBackend = ls.ml.create(
        title="Yolo Dart Keypoint",
        description="Yolo backend for dart keypoint detection ",
        project=project.id,
        url="http://yolo-ml-backend:9090",
    )
    print("Successfully connected to ML backend.")
except Exception as e:
    print(f"Failed to register ML backend: {e}")
