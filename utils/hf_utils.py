from huggingface_hub import HfApi, create_repo, login
import os

def upload_model_to_hub(
    local_folder,
    repo_id,
    token=None,
    private=True,
    commit_message="Upload model"
):
    """
    Upload a model to Hugging Face Hub
    
    Args:
        local_folder: Path to folder containing model files
        repo_id: Repository ID (username/model-name)
        token: HF token (optional if logged in)
        private: Whether repo should be private
        commit_message: Commit message
    
    Returns:
        repo_url: URL of the uploaded model
    """
    if token:
        login(token=token)
    
    # Create repository
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True
    )
    
    # Upload files
    api = HfApi()
    api.upload_folder(
        folder_path=local_folder,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message
    )
    
    repo_url = f"https://huggingface.co/{repo_id}"
    return repo_url


def check_model_exists(repo_id, token=None):
    """
    Check if a model exists on HF Hub
    
    Args:
        repo_id: Repository ID to check
        token: HF token (for private repos)
    
    Returns:
        bool: True if model exists
    """
    try:
        api = HfApi()
        api.model_info(repo_id, token=token)
        return True
    except:
        return False