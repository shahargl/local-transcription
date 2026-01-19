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

## Disclaimer

⚠️ **Note:** This script was primarily "vibe-coded" and is intended mainly for personal use. It has not been rigorously tested for production environments. Use at your own risk.

## Credits & Acknowledgements

This project relies heavily on the incredible work of [Ivrit.ai](https://www.ivrit.ai/en/ivrit-ai-2/).

Ivrit.ai is a non-profit organization dedicated to making Hebrew AI accessible to everyone. The default model used in this script (`ivrit-ai/whisper-large-v3-turbo-ct2`) is one of their state-of-the-art models. We deeply appreciate their contribution to the open-source community and their mission to advance Hebrew language technology.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

