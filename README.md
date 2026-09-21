# F5-TTS Spanish API

A **voice-cloning text-to-speech** model fine-tuned for **native Spanish** (Peninsular + Latin American accents), packaged as a clean, runnable local API.

This is a fixed, dependency-trimmed fork of [VoicePoweredAI_Spanish_v1](https://github.com/voicepowered-ai/VoicePoweredAI_Spanish_v1), which itself is a Spanish fine-tune of [F5-TTS](https://github.com/SWivid/F5-TTS).

## Why this model?

- **Native Spanish accent.** The base F5-TTS model is multilingual but its Spanish sounds like a non-native speaker. This fine-tune was trained on **218 hours of Spanish** (Peninsular, Argentine, Chilean, Colombian, Peruvian, Puerto Rican and Venezuelan accents), so it produces natural, native-sounding castellano.
- **Voice cloning.** You provide a short reference audio of *any* voice (male or female) and the model speaks your text in that voice, with a correct Spanish accent.

## ⚠️ Important: it clones, it has no voice catalog

This is a **voice-cloning** model, **not** a multi-voice catalog. There is no built-in list of "male voice #1 / female voice #2" to pick from. To get a male or female voice you must supply a **reference audio** of that voice (a few seconds of clean speech is enough). The model then reproduces the target text in that voice.

- Want a **female** voice? Provide a female reference recording.
- Want a **male** voice? Provide a male reference recording.
- The reference audio can be any `.wav`, `.mp3` or `.flac` file.

## Requirements

- macOS (Apple Silicon) or Linux with a GPU, or any machine with enough RAM/VRAM
- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/) (fast Python package manager) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- ~2 GB free disk for the model weights

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/dvdsosa/F5-TTS-Spanish-API.git
cd F5-TTS-Spanish-API
uv sync
```

`uv sync` creates a virtual environment and installs all dependencies (no `pip` needed).

### 2. Start the API

```bash
./scripts/serve.sh
```

On first run this downloads the Spanish checkpoint (~1.3 GB) and `vocab.txt` from Hugging Face into `./models/`, then starts the server on `http://127.0.0.1:8000`.

To pick a different port or device:

```bash
./scripts/serve.sh 9000 mps      # port 9000, Apple Silicon GPU
./scripts/serve.sh 8000 cuda     # NVIDIA GPU
./scripts/serve.sh 8000 cpu      # CPU only
```

### 3. Synthesize speech

With the API running, send a reference audio (the voice to clone) plus your Spanish text:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -F "ref_audio=@/path/to/female_voice.wav" \
  -F "text=Hola, este es un ejemplo de voz en castellano." \
  -o out.wav
```

Or use the included example script:

```bash
./examples/synthesize.sh "Hola, este es un ejemplo de voz en castellano." ref.wav out.wav
```

### 4. Use it from Python (no server)

```bash
uv run python examples/synthesize.py \
  --ref ref.wav \
  --text "Hola, esto es una prueba." \
  --out out.wav
```

## API reference

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

### `GET /models`

Returns model info and confirms it is a voice-cloning model (no catalog).

### `POST /v1/audio/speech`

Multipart form:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ref_audio` | file | ✅ | Reference audio (the voice to clone). `.wav`/`.mp3`/`.flac` |
| `text` | string | ✅ | Text to synthesize in Spanish |
| `ref_text` | string | ❌ | Transcript of the reference audio. If empty, auto-transcribed (needs `faster-whisper`) |
| `speed` | float | ❌ | Speech speed multiplier (default `1.0`) |
| `remove_silence` | bool | ❌ | Trim leading/trailing silence (default `false`) |

Returns a WAV file.

## Using it for video dubbing (EN → ES)

A typical pipeline:

1. **Transcribe** the English audio (e.g. with Whisper / faster-whisper).
2. **Translate** the transcript to Spanish.
3. **Synthesize** the Spanish text with this model, using a reference voice of your choice.

The API is OpenAI-compatible in spirit (`/v1/audio/speech`), so it can slot into tools that expect a TTS endpoint.

## License

- **Code:** Apache-2.0 (this repo).
- **Model weights:** the Hugging Face card for `jpgallegoar/F5-Spanish` lists `cc-by-nc-4.0` in its metadata (non-commercial), although the model README text claims CC0-1.0. **For personal/private use there is no issue.** If you plan commercial redistribution, clarify the license with the model author first.

## Credits

- [VoicePoweredAI_Spanish_v1](https://github.com/voicepowered-ai/VoicePoweredAI_Spanish_v1) — original Spanish fine-tune
- [F5-TTS](https://github.com/SWivid/F5-TTS) — base model
- [jpgallegoar/F5-Spanish](https://huggingface.co/jpgallegoar/F5-Spanish) — model weights on Hugging Face
