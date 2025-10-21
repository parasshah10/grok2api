import cloudinary
import cloudinary.uploader
import cloudinary.api
from .token import cloudinary_token_manager

class CloudinaryClient:
    def __init__(self):
        self.current_account_index = 0

    def get_next_account(self):
        accounts = cloudinary_token_manager.get_accounts()
        if not accounts:
            return None
        account = accounts[self.current_account_index]
        self.current_account_index = (self.current_account_index + 1) % len(accounts)
        return account

    def upload_video(self, video_path):
        account = self.get_next_account()
        if not account:
            raise Exception("No Cloudinary accounts configured")

        cloudinary.config(
            cloud_name=account.get("cloud_name"),
            api_key=account.get("api_key"),
            api_secret=account.get("api_secret")
        )

        response = cloudinary.uploader.upload_large(
            video_path,
            resource_type="video"
        )
        return response.get("secure_url")

cloudinary_client = CloudinaryClient()