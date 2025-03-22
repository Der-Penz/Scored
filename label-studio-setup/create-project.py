import argparse
import os
from label_studio_sdk import Client
from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIEW_FILE = os.path.join(SCRIPT_DIR, "view.yaml")

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=str, default='Scored', help='Name of the project')
args = parser.parse_args()

PROJECT_NAME = args.project

if not LABEL_STUDIO_URL or not API_KEY:
    print("Error: Missing LABEL_STUDIO_URL or API_KEY in .env")
    exit(1)

if not os.path.exists(VIEW_FILE):
    print(f"Error: {VIEW_FILE} not found in the current directory.")
    exit(1)

with open(VIEW_FILE, "r", encoding="utf-8") as file:
    label_config = file.read()

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)

project = ls.create_project(title=PROJECT_NAME, label_config=label_config)

print(f"Project '{PROJECT_NAME}' created successfully with labeling interface from '{VIEW_FILE}'.")
