# Welcome to the HackTheWorld(s) Hackathon! 🌍

## Before You Start

1. Connect to **Dalia** through SSH using the private key you received.
2. Open the GitHub Classroom link and join the classroom.
3. Retrieve the URL of your group's repository.

---

## Setup Instructions

### 1. Clone the Repository and Run the Setup Script

You can clone the repository anywhere, including your home directory. Replace `<YOUR_GROUP_REPOSITORY_URL>` with the URL provided by GitHub Classroom.

```bash
git clone <YOUR_GROUP_REPOSITORY_URL> eb_jepa
cd eb_jepa
bash setup.sh
```

The setup script will:

* Copy the repository to:

  ```text
  /lustre/work/pdl17890/$USER/eb_jepa
  ```

* Configure the project in that directory.

* Replace the original cloned folder with a one-line pointer `README` indicating the new location.

> ⚠️ **Important:** The setup script moves the repository to `/lustre/work`.
> **Do not be surprised if the repository seems to disappear from the directory where you originally cloned it.**
> This is expected behavior: everything must live under `/lustre/work`.

### 2. Move into the Work Copy

The pointer `README` gives you the exact path. By default, run:

```bash
cd /lustre/work/pdl17890/$USER/eb_jepa
```

### 3. Make the Environment Persistent and Verify the Setup

```bash
echo "source $(pwd)/env.sh" >> ~/.bashrc
source ~/.bashrc

sbatch slurm_test.sh
```

The `slurm_test.sh` job runs `pytest` on a GPU node to verify that the environment is working correctly.

---

## Environment and Cache Configuration

The `env.sh` script derives all paths from `$USER`.

All caches are stored under:

```text
$WORK/.cache
```

This includes:

* `uv`
* Hugging Face
* PyTorch
* Triton and `torch.compile`
* `pip`
* Weights & Biases

Nothing should be written to your home directory.

You can override the default work directory before running the setup:

```bash
export EBJEPA_WORK=/your/path
```

You can also specify the location of your datasets:

```bash
export EBJEPA_DSETS=/path/to/your/datasets
```

---

## ⚠️ Warning: Everything Must Live on `/lustre/work`

Everything must be stored under `/lustre/work`, **not in your home directory**.

The `/lustre/home` quota is small and may prevent:

* Git operations
* Virtual environment creation
* Package installation
* Model downloads
* Cache creation

You do not need to clone the repository directly into the correct directory: `setup.sh` automatically relocates the project to `/lustre/work`.

However, if you experience issues with the automatic relocation, clone the repository directly into your work directory:

```bash
cd /lustre/work/pdl17890/$USER

git clone <YOUR_GROUP_REPOSITORY_URL> eb_jepa
cd eb_jepa
bash setup.sh
```

## Troubleshooting

If the repository is no longer present in the directory where you initially cloned it, check:

```bash
cd /lustre/work/pdl17890/$USER/eb_jepa
```

This relocation is intentional and is required to avoid exceeding the home-directory quota.
