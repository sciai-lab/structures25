import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------
REPO_ID = "sciai-lab/structures25"

HUGGINGFACE_MODELS = {
    "QM9": "trained-on-qm9",
    "QMUGS": "trained-on-qmugs",
}

DEFAULT_DATA_DIR = Path(os.getenv("DFT_DATA", Path.home() / "dft_data"))
DEFAULT_MODELS_DIR = Path(os.getenv("DFT_MODELS", Path.home() / "dft_models"))
DEFAULT_STATISTICS_DIR = DEFAULT_MODELS_DIR / "dataset_statistics"


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user. "default" is the presumed answer if the
    user just hits <Enter>.         It must be "yes" (the default), "no" or None (meaning an answer
    is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def ask_path(prompt: str, default: Path) -> Path:
    """Prompt user for a directory path with a default."""
    user_input = input(f"{prompt} [{default}]: ").strip()
    chosen = Path(user_input) if user_input else default
    chosen.mkdir(parents=True, exist_ok=True)
    return chosen.resolve()


def download_model(model_name: str, repo_id: str, target_dir: Path):
    """Download a model from Hugging Face Hub into target directory."""
    print(f"⬇️  Downloading {model_name} from {repo_id}...")
    model_dir = target_dir / "train" / "runs"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        model_config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{model_name}/hparams.yaml",
            local_dir=model_dir,
        )
        ckpt_file = hf_hub_download(
            repo_id=repo_id,
            filename=f"{model_name}/{model_name}.ckpt",
            local_dir=model_dir,
        )
        ckpt_dir = model_dir / f"{model_name}" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        Path(ckpt_file).rename(ckpt_dir / "last.ckpt")
        print(f"✅ {model_name} downloaded to {model_dir}")
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")


def download_dataset_statistics(repo_id: str, target_dir: Path):
    """Download dataset statistics from Hugging Face Hub into target directory."""
    print(f"⬇️  Downloading statistics from {repo_id}...")
    try:
        print(f"Saving to {target_dir}")
        # Download only a subdirectory (e.g., "data/train")
        prefix = "sciai-test-mol/dataset_statistics/dataset_statistics_labels_no_basis_transforms_e_kin_plus_xc.zarr/"
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                prefix + suffix for suffix in ["*0", "*.zarray", "*.zattrs", "*.zgroup"]
            ],
            local_dir=target_dir,
        )
        print(f"✅ Dataset statistics downloaded to {target_dir}")
    except Exception as e:
        print(f"❌ Failed to download dataset statistics: {e}")


def ask_download_models(models_dir: Path, repo_id: str = REPO_ID):
    """Ask user if they want to download models."""
    question = "Would you like to download the QM9 and QMUGS models from Hugging Face?"
    answer = query_yes_no(question, default="yes")
    if answer:
        for key, name in HUGGINGFACE_MODELS.items():
            download_model(name, repo_id, models_dir)
    else:
        print("Skipping model downloads.")


def ask_download_dataset_statistics(statistics_dir: Path, repo_id: str = REPO_ID):
    """Ask user if they want to download models."""
    question = "Would you like to download dataset statistics from Hugging Face for the SAD guess?"
    answer = query_yes_no(question, default="yes")
    if answer:
        download_dataset_statistics(repo_id, statistics_dir)
    else:
        print("Skipping dataset statistics downloads.")


def main():
    """Main setup function."""
    print("🚀 MLDFT Package Setup\n--------------------")

    data_dir = ask_path("Enter data directory path", DEFAULT_DATA_DIR)
    models_dir = ask_path("Enter models directory path", DEFAULT_MODELS_DIR)
    statistics_dir = ask_path("Enter models directory path", DEFAULT_STATISTICS_DIR)

    print("\n📋 Using the following paths for setup:")
    print(f"  - DFT_DATA       = {data_dir}")
    print(f"  - DFT_MODELS     = {models_dir}")
    print(f"  - DFT_STATISTICS = {statistics_dir}")

    print("\nTo set these environment variables in your current shell, run:")
    print(f"export DFT_DATA='{data_dir}'")
    print(f"export DFT_MODELS='{models_dir}'")
    print(f"export DFT_STATISTICS='{statistics_dir}'")
    print("\nTo make them permanent, add the above lines to your ~/.zshrc file.")

    ask_download_models(models_dir, repo_id=REPO_ID)
    ask_download_dataset_statistics(statistics_dir, repo_id=REPO_ID)

    print("\n✅ Setup complete! 🎉")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        sys.exit(1)
