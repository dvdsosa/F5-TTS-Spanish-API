"""Example: synthesize Spanish speech from a reference voice using the F5TTS class directly.

This shows the voice-cloning workflow: you provide a short reference audio of the
voice you want to clone (male or female), and the model speaks the target text in
that voice with a native Spanish accent.

Usage:
    uv run python examples/synthesize.py \
        --ref ref.wav --text "Hola, esto es una prueba." --out out.wav
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from f5_tts.api import F5TTS  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CKPT = MODEL_DIR / "model_1200000.safetensors"
VOCAB = MODEL_DIR / "vocab.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Spanish speech by cloning a reference voice.")
    parser.add_argument("--ref", required=True, help="Reference audio (the voice to clone), wav/mp3/flac")
    parser.add_argument("--text", required=True, help="Text to synthesize in Spanish")
    parser.add_argument("--ref-text", default="", help="Optional transcript of the reference audio")
    parser.add_argument("--out", default="out.wav", help="Output wav path")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    if not CKPT.exists() or not VOCAB.exists():
        sys.exit("Model not found. Run ./scripts/serve.sh once to download it, or set CKPT/VOCAB paths.")

    print("Loading model ...")
    tts = F5TTS(model_type="F5-TTS", ckpt_file=str(CKPT), vocab_file=str(VOCAB), vocoder_name="vocos")

    print("Synthesizing ...")
    wav, sr, _ = tts.infer(
        ref_file=args.ref,
        ref_text=args.ref_text,
        gen_text=args.text,
        speed=args.speed,
        file_wave=args.out,
    )
    print(f"Saved to {args.out} ({sr} Hz, {len(wav) / sr:.1f}s)")


if __name__ == "__main__":
    main()
