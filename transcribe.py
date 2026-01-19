#!/usr/bin/env python3
"""
Transcription Tool
==================

A command-line tool to transcribe audio and video files using WhisperX.
Supports batch processing, speaker diarization, and multiple media formats.
"""

import argparse
import gc
import json
import logging
import os
import sys
import traceback
import warnings
from typing import List, Optional, Tuple, Set, Union, Dict, Any

# Force unbuffered output for real-time progress display
os.environ["PYTHONUNBUFFERED"] = "1"

from dotenv import load_dotenv

# Filter warnings immediately to keep output clean
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends has been deprecated.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded your loaded checkpoint.*")

# Constants
DEFAULT_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
DEFAULT_BATCH_SIZE = 16
DEFAULT_COMPUTE_TYPE_CUDA = "float16"
DEFAULT_COMPUTE_TYPE_CPU = "int8"
VALID_EXTENSIONS: Set[str] = {'.mkv', '.mp4', '.m4a', '.mov', '.amr', '.wav', '.mp3', '.flac', '.opus'}


def patch_torch() -> None:
    """
    Monkey-patch torch.load to default weights_only=False.
    
    This is a workaround for PyTorch 2.6+ where weights_only=True became the default,
    which breaks loading older checkpoints used by pyannote/whisperx.
    """
    import torch
    original_load = torch.load

    def safe_load(*args, **kwargs):
        # Force weights_only=False even if True was passed or if it's the default
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)

    torch.load = safe_load


def setup_logging() -> None:
    """Configure logging for the application and suppresses noisy libraries."""
    # Configure logging before other imports to prevent basicConfig stealing control
    logging.basicConfig(level=logging.ERROR)
    
    # Suppress verbose logs from dependencies
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("pyannote").setLevel(logging.ERROR)
    logging.getLogger("speechbrain").setLevel(logging.ERROR)
    logging.getLogger("whisperx").setLevel(logging.ERROR)


# Import heavy libraries after basic setup to avoid delays/side-effects if they aren't needed immediately
setup_logging()
patch_torch()
load_dotenv()

import torch
import whisperx


class Transcriber:
    """
    Handles model loading, transcription, alignment, and diarization.
    """

    def __init__(self, model_size: str, device: str, language: Optional[str] = None):
        """
        Initialize the Transcriber. 

        Args:
            model_size (str): Name of the Whisper model to use.
            device (str): Device to run on ('cuda' or 'cpu').
            language (str, optional): Target language code (e.g., 'he', 'en'). 
                                      If None, auto-detection is used.
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self.compute_type = DEFAULT_COMPUTE_TYPE_CUDA if device == "cuda" else DEFAULT_COMPUTE_TYPE_CPU
        
        self.model = None
        self.align_model = None
        self.align_metadata = None

    def load_model(self) -> None:
        """Loads the main Whisper transcription model."""
        print(f"Loading transcription model '{self.model_size}' on {self.device}...")
        try:
            self.model = whisperx.load_model(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type, 
                language=self.language
            )
            print("Model loaded successfully.")
        except Exception as e:
            sys.exit(f"Failed to load model: {e}")

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribes an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            dict: The transcription result containing segments and detected language.
        """
        if not self.model:
            self.load_model()

        print(f"Loading audio: {os.path.basename(audio_path)}...", flush=True)
        audio = whisperx.load_audio(audio_path)
        
        # Calculate audio duration for progress display
        audio_duration = len(audio) / 16000  # whisperx uses 16kHz sample rate
        print(f"Transcribing {audio_duration:.1f}s of audio...", flush=True)
        
        try:
            result = self.model.transcribe(
                audio, 
                batch_size=DEFAULT_BATCH_SIZE, 
                language=self.language, 
                print_progress=True
            )
        except TypeError:
            print("Warning: print_progress=True failed, trying default.", flush=True)
            result = self.model.transcribe(
                audio, 
                batch_size=DEFAULT_BATCH_SIZE, 
                language=self.language
            )
        
        return result

    def align(self, result: Dict[str, Any], audio_path: str) -> Dict[str, Any]:
        """
        Aligns the transcription result to the audio for precise timestamps.

        Args:
            result (dict): The result from self.transcribe().
            audio_path (str): Path to the audio file.

        Returns:
            dict: The aligned result.
        """
        print("Aligning transcript...")
        try:
            # Load alignment model if not loaded or if language changed (rare in batch if same lang)
            # For simplicity, we assume one align model is enough if we stick to one language,
            # but we check the language of the result just in case.
            language_code = result["language"]
            
            if self.align_model is None:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=language_code, 
                    device=self.device
                )
            
            # Reload audio for alignment (whisperx.align expects different audio object sometimes? 
            # No, it expects numpy array from load_audio. We can just reload to be safe or pass it if cached.)
            # Optimally, we shouldn't reload if we just loaded it, but keeping it simple for now.
            audio = whisperx.load_audio(audio_path)

            result = whisperx.align(
                result["segments"], 
                self.align_model, 
                self.align_metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            return result
        except Exception as e:
            print(f"Warning: Alignment failed: {e}. Returning unaligned result.")
            return result

    def diarize(self, result: Dict[str, Any], audio_path: str, hf_token: str) -> Dict[str, Any]:
        """
        Performs speaker diarization on the aligned result.

        Args:
            result (dict): The result from self.align() (or self.transcribe()).
            audio_path (str): Path to the audio file.
            hf_token (str): Hugging Face authentication token.

        Returns:
            dict: The result with speaker labels assigned.
        """
        print("Starting speaker diarization...", flush=True)
        try:
            # We import here to avoid loading the pipeline if not needed
            from whisperx.diarize import DiarizationPipeline
            
            diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=self.device)
            audio = whisperx.load_audio(audio_path)
            diarize_segments = diarize_model(audio)
            
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print("Diarization complete.", flush=True)
            return result
        except Exception as e:
            print(f"Diarization failed: {e}")
            traceback.print_exc()
            print("Proceeding with transcription only.")
            return result

    def cleanup(self):
        """Free up memory."""
        if self.model:
            del self.model
            self.model = None
        if self.align_model:
            del self.align_model
            self.align_model = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


def get_files_to_process(path: str) -> List[str]:
    """
    Scans a directory or file path for valid media files.
    """
    files_to_process = []
    if os.path.isdir(path):
        print(f"Scanning directory '{path}' for media files...")
        try:
            for f in sorted(os.listdir(path)):
                if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS:
                    files_to_process.append(os.path.join(path, f))
        except Exception as e:
            sys.exit(f"Error scanning directory: {e}")
        
        if not files_to_process:
            print(f"No media files found in '{path}' with extensions: {', '.join(VALID_EXTENSIONS)}")
            sys.exit(0)
        print(f"Found {len(files_to_process)} files to process.")
    else:
        if not os.path.exists(path):
            sys.exit(f"Error: File not found at '{path}'")
        files_to_process = [path]
    return files_to_process


def format_time(seconds: float) -> str:
    """Formats seconds into [HH:MM:SS]."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"[{int(h):02d}:{int(m):02d}:{int(s):02d}]"


def format_output(result: Dict[str, Any], include_speakers: bool = False, include_timestamps: bool = False) -> str:
    """Formats the transcription result into a string."""
    output_lines = []
    for segment in result["segments"]:
        text = segment["text"].strip()
        parts = []
        
        if include_timestamps:
            # defined per segment start
            start_time = segment.get("start", 0.0)
            parts.append(format_time(start_time))

        if include_speakers:
            speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
            parts.append(f"[{speaker}]:")
        
        parts.append(text)
        output_lines.append(" ".join(parts))
        
    return "\n".join(output_lines)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files using WhisperX. Supports single file or batch processing."
    )
    parser.add_argument(
        "media_file", 
        help="Path to the media file or folder to transcribe"
    )
    parser.add_argument(
        "--diarize", 
        action="store_true", 
        help="Enable speaker diarization (requires HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--diarize-only",
        action="store_true",
        help="Only run diarization on existing transcription (requires .json result file from previous run)"
    )
    parser.add_argument(
        "--reformat",
        action="store_true",
        help="Just regenerate .txt output from existing .json (no transcription or diarization)"
    )
    parser.add_argument(
        "--model", 
        default=DEFAULT_MODEL, 
        help=f"Whisper model size/name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--language", 
        default="he", 
        help="Language code for transcription (default: 'he'). Use 'auto' or leave empty for detection."
    )
    parser.add_argument(
        "--device", 
        default=None, 
        help="Compute device (cuda/cpu). Defaults to cuda if available."
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in the output (format [HH:MM:SS])"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Handle Language
    language = None if args.language in ["auto", "None"] else args.language

    # Scan files
    files = get_files_to_process(args.media_file)

    # Initialize Transcriber
    # Note: If diarizing, we might want to load/unload models per file to avoid VRAM OOM,
    # but the Transcriber class can handle that logic if we chose to. 
    # For now, we keep the model loaded if possible for speed, unless diarization triggers memory pressure.
    transcriber = Transcriber(model_size=args.model, device=device, language=language)

    # Load model once if not diarizing (heuristic optimization)
    # Skip model loading entirely for --diarize-only or --reformat mode
    if not args.diarize and not args.diarize_only and not args.reformat:
        transcriber.load_model()

    for i, media_file in enumerate(files):
        print(f"\n--- Processing file {i+1}/{len(files)}: '{media_file}' ---")
        base_name = os.path.splitext(media_file)[0]
        json_file = f"{base_name}.json"

        try:
            # Handle --reformat mode (just regenerate .txt from existing .json)
            if args.reformat:
                if not os.path.exists(json_file):
                    print(f"❌ Error: No existing transcription found at '{json_file}'")
                    print("   Run transcription first without --reformat")
                    continue
                
                print(f"Loading existing transcription from '{json_file}'...")
                with open(json_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
                
                # Check if result has speaker info
                has_speakers = any("speaker" in seg for seg in result.get("segments", []))
                
                # Save Output with new formatting options
                txt_content = format_output(
                    result, 
                    include_speakers=has_speakers, 
                    include_timestamps=args.timestamps
                )
                
                output_file = f"{base_name}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(txt_content)
                print(f"✅ Success! Reformatted output saved to '{output_file}'")
                continue
            
            # Handle --diarize-only mode
            if args.diarize_only:
                if not os.path.exists(json_file):
                    print(f"❌ Error: No existing transcription found at '{json_file}'")
                    print("   Run transcription first without --diarize-only")
                    continue
                
                print(f"Loading existing transcription from '{json_file}'...")
                with open(json_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
                
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    print("❌ Error: HF_TOKEN not set. Required for diarization.")
                    continue
                
                result = transcriber.diarize(result, media_file, hf_token)
            else:
                # 1. Transcribe
                # If diarizing, ensuring clean slate might be safer for VRAM
                if args.diarize:
                    # Re-init/ensure loaded
                    if not transcriber.model:
                         transcriber.load_model()
                
                result = transcriber.transcribe(media_file)

                # 2. Align
                result = transcriber.align(result, media_file)
                
                # Save intermediate result as JSON for potential --diarize-only later
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"Intermediate result saved to '{json_file}'")

                # 3. Diarize (Optional)
                if args.diarize:
                    # Aggressive cleanup before diarization to save VRAM
                    # We delete the whisper model before loading the pyannote pipeline
                    transcriber.cleanup()
                    
                    hf_token = os.environ.get("HF_TOKEN")
                    if not hf_token:
                        print("Warning: HF_TOKEN not set. Skipping diarization.")
                    else:
                        result = transcriber.diarize(result, media_file, hf_token)

            # 4. Save Output
            txt_content = format_output(
                result, 
                include_speakers=args.diarize or args.diarize_only, 
                include_timestamps=args.timestamps
            )
            
            output_file = f"{base_name}.txt"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(txt_content)
            print(f"✅ Success! Transcription saved to '{output_file}'")

        except Exception as e:
            print(f"❌ Critical error processing '{media_file}': {e}")
            traceback.print_exc()
        finally:
            # If we were diarizing, we already cleaned up. 
            # If not, we keep the model for the next file.
            if args.diarize:
                transcriber.cleanup()

    print("\nAll files processed.")


if __name__ == "__main__":
    main()