import kagglehub
from kagglehub.config import get_kaggle_credentials

kagglehub.login()
username, _ = get_kaggle_credentials()

# Upload model files - adjust paths and framework accordingly
kagglehub.model_upload(f'{username}/my_model/pyTorch/2b', 'path/to/local/model/files', 'Apache 2.0')
