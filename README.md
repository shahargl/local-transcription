# Local Hebrew Transcription Script

This project uses `faster-whisper` and the [Ivrit.ai](https://www.ivrit.ai/en/ivrit-ai-2/) model to transcribe audio/video files.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) (An extremely fast Python package installer and resolver)

## Setup

1.  Make sure `uv` is installed.
2.  Clone this repository (if not already done).
3.  The environment and dependencies will be automatically handled by `uv` when you run the script.
4.  Create a `.env` file in the project root to store your configuration:

    ```bash
    HF_TOKEN=your_hugging_face_token
    ```

    > **Note:** The `HF_TOKEN` is required for **speaker diarization** (the process of separating audio into different speakers). It is used **only** to verify that you have accepted the user agreement for the `pyannote/speaker-diarization-3.1` model on Hugging Face. No sensitive data is accessed or shared. You must accept the agreement here: https://huggingface.co/pyannote/speaker-diarization-3.1

## Usage

To run the transcription script, use `uv run`:

### Single File
```bash
uv run transcribe.py path/to/media_file
```

### Folder (Batch Processing)
To transcribe all media files in a directory:
```bash
uv run transcribe.py path/to/folder
```

Supported extensions: `.mkv`, `.mp4`, `.m4a`, `.mov`, `.amr`, `.wav`, `.mp3`, `.flac`, `.opus`.

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--diarize` | Enable speaker diarization (requires `HF_TOKEN`) | False |
| `--timestamps` | Include timestamps in output (`[HH:MM:SS]`) | False |
| `--model` | Whisper model size/name | `ivrit-ai/whisper-large-v3-turbo-ct2` |
| `--language` | Language code (use `auto` for detection) | `he` |
| `--device` | Compute device (`cuda` or `cpu`) | Auto-detected |

Simple example:
```bash
uv run transcribe.py interview.mp3
```

Example with diarization (different speakers identification):
```bash
uv run transcribe.py --diarize interview.mp3
```

**Note:** Transcription outputs are saved as `.txt` files in the same directory as the source media.

### Re-processing Options

| Flag | Description |
|------|-------------|
| `--diarize-only` | Re-run diarization on existing transcription (requires `.json` from previous run) |
| `--reformat` | Regenerate `.txt` from existing `.json` without any processing |

These are useful when you want to:
- Add timestamps to an existing transcription: `uv run transcribe.py --reformat --timestamps file.mp3`
- Re-run diarization with different settings: `uv run transcribe.py --diarize-only file.mp3`

## Running on Google Colab

You can run this on Google Colab to leverage free GPU acceleration. Here's how:

### 1. Setup Cell

```python
# Install uv and clone the repo
!curl -LsSf https://astral.sh/uv/install.sh | sh
!git clone https://github.com/joroizin/local-transcription.git
%cd local-transcription

# Create a virtual environment with Python 3.13
!~/.local/bin/uv venv --python 3.13

# Install dependencies (whisperx from git to avoid PyPI conflicts)
!~/.local/bin/uv pip install -p .venv/bin/python \
    git+https://github.com/m-bain/whisperx.git \
    python-dotenv

# Set your HuggingFace token (required for diarization)
import os
os.environ["HF_TOKEN"] = "your_token_here"  # Get from https://huggingface.co/settings/tokens
os.environ["MPLBACKEND"] = "agg"  # Fix matplotlib backend for Colab
```

### 2. Upload Your Audio File

```python
from google.colab import files
uploaded = files.upload()  # Select your audio file
```

Or mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
# Then use: /content/drive/MyDrive/path/to/file.mp3
```

### 3. Run Transcription

```python
# Basic transcription
!.venv/bin/python transcribe.py your_file.mp3

# With speaker diarization and timestamps
!.venv/bin/python transcribe.py --diarize --timestamps your_file.mp3
```

### 4. Download Results

```python
from google.colab import files
files.download('your_file.txt')
files.download('your_file.json')  # Contains full data with timestamps
```

## Disclaimer

⚠️ **Note:** This script was primarily "vibe-coded" and is intended mainly for personal use. It has not been rigorously tested for production environments. Use at your own risk.

## Credits & Acknowledgements

This project relies heavily on the incredible work of [Ivrit.ai](https://www.ivrit.ai/en/ivrit-ai-2/).

Ivrit.ai is a non-profit organization dedicated to making Hebrew AI accessible to everyone. The default model used in this script (`ivrit-ai/whisper-large-v3-turbo-ct2`) is one of their state-of-the-art models. We deeply appreciate their contribution to the open-source community and their mission to advance Hebrew language technology.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

