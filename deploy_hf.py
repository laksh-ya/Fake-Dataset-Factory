"""Create the public model repository and free Hugging Face Space."""

import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).parent
CHECKPOINTS = [
    "notebooks/outputs/02_dcgan/checkpoints/checkpoint_epoch_50.pt",
    "notebooks/outputs/03_wgan_gp/checkpoints/checkpoint_epoch_50.pt",
    "notebooks/outputs/04_vqvae/checkpoints/checkpoint_epoch_50.pt",
    "notebooks/outputs/05_ddpm/checkpoints/checkpoint_epoch_50.pt",
    "notebooks/outputs/06_flow_matching_v2/checkpoints/checkpoint_epoch_100.pt",
]

api = HfApi()
owner = os.getenv("HF_OWNER") or api.whoami()["name"]
model_repo = f"{owner}/fake-dataset-factory-weights"
space_repo = f"{owner}/fake-dataset-factory"

api.create_repo(model_repo, repo_type="model", private=False, exist_ok=True)
for relative_path in CHECKPOINTS:
    source = ROOT / relative_path
    if not source.is_file():
        raise FileNotFoundError(source)
    print(f"Uploading unchanged checkpoint: {relative_path}")
    api.upload_file(
        path_or_fileobj=source,
        path_in_repo=relative_path,
        repo_id=model_repo,
        repo_type="model",
    )

if not api.repo_exists(space_repo, repo_type="space"):
    raise RuntimeError(
        f"Existing Space {space_repo} was not found; refusing to create a paid Space."
    )
operations = [
    CommitOperationAdd("app.py", ROOT / "app.py"),
    CommitOperationAdd("ui.py", ROOT / "ui.py"),
    CommitOperationAdd("requirements.txt", ROOT / "requirements.txt"),
    CommitOperationAdd("README.md", ROOT / "space_README.md"),
    CommitOperationAdd(
        "notebooks/outputs/01_stable_diffusion/metrics.json",
        ROOT / "notebooks/outputs/01_stable_diffusion/metrics.json",
    ),
]
api.add_space_variable(space_repo, "WEIGHTS_REPO_ID", model_repo)
api.create_commit(
    repo_id=space_repo, repo_type="space", operations=operations,
    commit_message="Deploy free CPU Gradio app",
)
print(f"Deployed: https://huggingface.co/spaces/{space_repo}")
