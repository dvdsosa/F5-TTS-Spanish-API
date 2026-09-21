"""Local voice-cloning TTS API for the Spanish F5-TTS model.

Run with:
    uv run f5-tts-spanish-api --ckpt <path-to-model.safetensors> --vocab <path-to-vocab.txt>

Or via the convenience script:
    ./scripts/serve.sh
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from f5_tts.api import F5TTS

app = FastAPI(title="F5-TTS Spanish API", version="1.0.0")

# Global model handle, set at startup via --ckpt / --vocab
_tts: F5TTS | None = None
_ckpt: str = ""
_vocab: str = ""


def _load_model(ckpt: str, vocab: str, device: str | None = None) -> F5TTS:
    """Instantiate the F5-TTS model with the Spanish checkpoint."""
    return F5TTS(
        model_type="F5-TTS",
        ckpt_file=ckpt,
        vocab_file=vocab,
        vocoder_name="vocos",
        device=device,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _tts is not None}


@app.get("/models")
def models() -> dict:
    return {
        "model": "F5-TTS (Spanish fine-tune)",
        "checkpoint": os.path.basename(_ckpt) if _ckpt else None,
        "vocab": os.path.basename(_vocab) if _vocab else None,
        "voice_cloning": True,
        "voice_catalog": False,
        "note": "This is a voice-cloning model. Provide a reference audio of the voice you want to clone.",
    }


@app.post("/v1/audio/speech")
async def synthesize(
    ref_audio: UploadFile = File(..., description="Reference audio (the voice to clone), wav/mp3/flac"),
    text: str = Form(..., description="Text to synthesize in Spanish"),
    ref_text: str = Form("", description="Optional transcript of the reference audio. If empty, it is auto-transcribed (requires faster-whisper)."),
    speed: float = Form(1.0, description="Speech speed multiplier"),
    remove_silence: bool = Form(False, description="Trim leading/trailing silence"),
) -> Response:
    if _tts is None:
        raise HTTPException(503, "Model not loaded. Start the server with --ckpt and --vocab.")

    # Save the uploaded reference audio to a temp file
    suffix = Path(ref_audio.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(ref_audio.file.read())
        ref_path = tmp.name

    try:
        wav, sr, _ = _tts.infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=text,
            speed=speed,
            remove_silence=remove_silence,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Inference failed: {exc}") from exc
    finally:
        os.unlink(ref_path)

    # Encode to WAV bytes
    buf = tempfile.SpooledTemporaryFile()
    sf.write(buf, wav, sr, format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Spanish F5-TTS model as a local API.")
    parser.add_argument("--ckpt", required=True, help="Path to the Spanish model checkpoint (.safetensors)")
    parser.add_argument("--vocab", required=True, help="Path to vocab.txt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    args = parser.parse_args()

    global _tts, _ckpt, _vocab
    _ckpt, _vocab = args.ckpt, args.vocab
    print(f"Loading model from {args.ckpt} ...")
    _tts = _load_model(args.ckpt, args.vocab, args.device)
    print(f"Model loaded. Serving on http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
