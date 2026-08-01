import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# --- Credentials & Config Setup ---
# Edit these values, then run every cell top to bottom.
GITHUB_USERNAME   = "Jeyakumag"       # Your GitHub username
HF_USERNAME       = "JeyAIML"         # Your Hugging Face username 
REPO_NAME         = "ModelProject"    # Repository name
COLAB_SECRET_NAME = "GH_TOKEN"        # Name of the secret in Colab

# Constructing repository identity using HF username
# This creates: "JeyAIML/ModelProject"
repo_id = f"{HF_USERNAME}/{REPO_NAME}"
repo_type = "dataset"

# Fetch token safely from Colab Secrets
try:
    from google.colab import userdata
    token = userdata.get(COLAB_SECRET_NAME)
except (ImportError, Exception):
    # Fallback to environment variable if running outside Colab
    token = os.getenv("HF_TOKEN")

# Initialize API client
api = HfApi(token=token)

# Check if the repository exists, create it if missing
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Skipping creation.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=token)
    print(f"Dataset '{repo_id}' created successfully.")

# Upload the contents of the folder
print(f"Uploading folder to Hugging Face...")
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Upload complete!")