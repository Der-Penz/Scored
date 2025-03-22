import os
import httpx
from label_studio_sdk import Client
from label_studio_sdk.core import SyncClientWrapper
from label_studio_sdk.ml.client import MlClient
from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
PROJECT_NAME = "Scored" # Change the name if needed

ML_BACKEND_URL = "http://yolo-ml-backend:9090"
ML_BACKEND_TITLE = "YOLO ML Backend"

if not LABEL_STUDIO_URL or not API_KEY:
    print("Error: Missing LABEL_STUDIO_URL or API_KEY in .env")
    exit(1)

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)

projects = ls.list_projects()
project = next((p for p in projects if p.get_params()["title"] == PROJECT_NAME), None)

if not project:
    print(f"Error: Project '{PROJECT_NAME}' not found.")
    exit(1)

print(f"Found project '{PROJECT_NAME}' with ID: {project.id}")

httpx_client = httpx.Client()
sync = SyncClientWrapper(api_key=API_KEY, base_url=LABEL_STUDIO_URL, httpx_client=httpx_client)

ml_backend = MlClient(client_wrapper=sync)

try:
    ml_backend.create(url=ML_BACKEND_URL, project=project.id, title=ML_BACKEND_TITLE)
    print(f"Successfully registered ML backend '{ML_BACKEND_URL}' to project '{PROJECT_NAME}'.")
except Exception as e:
    print(f"Failed to register ML backend: {e}")
