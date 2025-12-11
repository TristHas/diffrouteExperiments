from src.config import DATA_ROOT as root
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="prediction2/diffroute_exp",  # or any other dataset id
    repo_type="dataset",
    local_dir=root,      # where to put the files
)
print("Downloaded to:", local_dir)