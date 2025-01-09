# app/utils.py
import huggingface_hub
from huggingface_hub import HfApi, create_repo
from app.core.config import Config

class HuggingFaceManager:
    def __init__(self):
        self.api = HfApi()
        self.config = Config()
        huggingface_hub.login(token=self.config.HUGGINGFACE_API_TOKEN)

    def init_space(self, space_name):
        """Initialize a new Hugging Face Space"""
        try:
            create_repo(
                space_name,
                space_sdk="streamlit",
                private=False
            )
            return True
        except Exception as e:
            print(f"Error creating space: {e}")
            return False

    def upload_files(self, space_name, file_paths):
        """Upload files to the Space"""
        try:
            for file_path in file_paths:
                self.api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path,
                    repo_id=space_name,
                    repo_type="space"
                )
            return True
        except Exception as e:
            print(f"Error uploading files: {e}")
            return False