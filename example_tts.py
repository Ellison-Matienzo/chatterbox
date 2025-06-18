from __future__ import annotations

"""Text-to-speech CLI

Reads ``input.txt`` and converts it to speech. Control how many lines are
concatenated **per TTS request** with ``--chunk-size/-n``:

* ``-n 1`` (default)  → one request per non-blank line (best prosody)
* ``-n 3``            → three lines per request
* ``-n 0`` or ``-n -1`` → the entire file in a single request

The resulting waveforms are concatenated and written to one ``*.wav`` file.
If a custom voice prompt is used, the output filename stem is that voice's
base name; otherwise it is ``test-default``.
"""

import argparse
import sys
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, List

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Import the text formatting function from the other script
from split_sentences import format_text_into_lines


# ── Utilities ────────────────────────────────────────────────────────────────────

def best_device() -> torch.device:
    """Return the first available device in CUDA ▸ MPS ▸ CPU order."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():  # pragma: no cover  (macOS only)
        return torch.device("mps")
    return torch.device("cpu")


def read_input_lines(path: Path = Path("input.txt")) -> List[str]:
    """Return a list of non-blank, stripped lines from *path*, or exit if bad."""
    if not path.exists():
        sys.exit(f"Error: '{path}' not found.")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        sys.exit(f"Error: '{path}' contains no usable text.")
    return lines


def choose_voice(voices_dir: Path = Path("voices")) -> Path | None:
    """Interactive picker for a voice-prompt ``.wav`` file inside *voices_dir*."""
    wav_files = sorted(voices_dir.glob("*.wav"))
    if not wav_files or input("Use a custom voice prompt? [y/N] ").strip().lower() != "y":
        return None

    print("\nAvailable voice prompts:")
    for i, wav in enumerate(wav_files, start=1):
        print(f"{i}) {wav.name}")

    while True:
        choice = input(f"Select 1-{len(wav_files)} (Enter to cancel): ").strip()
        if not choice:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(wav_files):
            return wav_files[int(choice) - 1]
        print("Invalid selection, try again.")


def timestamped_filename(stem: str, *, ext: str = ".wav") -> str:
    """Return ``stem-YYYY-MM-DD-hh-mm-ss.ext`` in local time."""
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{stem}-{ts}{ext}"


# ── Core synthesis helpers ───────────────────────────────────────────────────────

def _chunks(xs: List[str], n: int) -> Iterator[List[str]]:
    """Yield *n*-sized chunks from *xs*.  ``n<=0`` → the whole list once."""
    if n <= 0:
        yield xs
    else:
        it = iter(xs)
        while chunk := list(islice(it, n)):
            yield chunk


def synthesize(
    lines: Iterable[str],
    out_path: Path,
    *,
    chunk_size: int = 1,
    audio_prompt: Path | None = None,
    device: torch.device | None = None,
) -> None:
    """Generate speech for *lines* (batched by *chunk_size*) and save to *out_path*."""

    device = device or best_device()
    print(f"\nUsing device: {device}\n")

    tts = ChatterboxTTS.from_pretrained(device=device)
    segments: List[torch.Tensor] = []

    for idx, chunk in enumerate(_chunks(list(lines), chunk_size), 1):
        text_block = "\n".join(chunk)
        preview = text_block.replace("\n", " ⏎ ")[:70]
        print(f"[TTS] ({idx}) → '{preview}{'…' if len(text_block) > 70 else ''}'")
        segment = tts.generate(text_block, audio_prompt_path=str(audio_prompt) if audio_prompt else None)
        segments.append(segment)

    if not segments:
        sys.exit("Nothing to synthesize – all lines were blank.")

    waveform = torch.cat(segments, dim=1)  # concatenate time dimension
    ta.save(str(out_path), waveform, tts.sr)
    print(f"Saved: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate speech from input.txt.")
    parser.add_argument(
        "-n", "--chunk-size",
        type=int,
        default=1,
        metavar="N",
        help="Number of input lines to concatenate per TTS call (default: 1; 0 or <0 → all lines).",
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="auto",
        help='Device to use, e.g., "cuda:0", "cpu", or "auto" for automatic selection (default: auto).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- MODIFICATION: Pre-format input.txt ---
    input_path = Path("input.txt")
    print(f"Reading and formatting '{input_path}'...")
    try:
        original_text = input_path.read_text(encoding="utf-8")
        if not original_text.strip():
            sys.exit(f"Error: '{input_path}' contains no usable text.")

        formatted_text = format_text_into_lines(original_text)
        input_path.write_text(formatted_text, encoding="utf-8")
        print(f"'{input_path}' has been re-formatted for optimal TTS.")

    except FileNotFoundError:
        sys.exit(f"Error: '{input_path}' not found.")
    # --- END MODIFICATION ---

    lines = read_input_lines(input_path)
    voice = choose_voice()

    # --- DEVICE SELECTION LOGIC ---
    if args.device == "auto":
        device = best_device()
    else:
        device = torch.device(args.device)

    # Add validation for CUDA devices
    if device.type == "cuda":
        if not torch.cuda.is_available():
            sys.exit(f"Error: CUDA is not available, but '{device}' was requested.")

        # device.index is None for 'cuda', which implies 'cuda:0'
        device_id = device.index if device.index is not None else 0
        if device_id >= torch.cuda.device_count():
            sys.exit(
                f"Error: Invalid CUDA device ID: {device_id}. "
                f"Available devices: {torch.cuda.device_count()}."
            )

    # Use voice file's stem (without extension) if provided, else fallback
    stem = voice.stem.replace(" ", "_") if voice else "test-default"
    out_file = Path(timestamped_filename(stem))

    synthesize(
        lines,
        out_file,
        chunk_size=args.chunk_size,
        audio_prompt=voice,
        device=device,
    )


if __name__ == "__main__":
    main()
