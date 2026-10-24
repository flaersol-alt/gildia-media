"""Локальная озвучка русского текста нейромоделью Silero v4 (без отправки текста в сеть).

Установка (один раз):
    pip install torch numpy
    # нужен ffmpeg в PATH для сборки mp3

Запуск:
    python voice_silero.py text.txt podcast.mp3
    python voice_silero.py text.txt podcast.mp3 baya

Голоса: xenia, baya (женские), aidar, eugene (мужские). Модель скачивается
один раз с models.silero.ai, дальше работает офлайн. Скачанные веса лежат
в ~/.cache/torch/hub.
"""
import os
import re
import subprocess
import sys
import tempfile
import wave

import torch

SAMPLE_RATE = 48000
MAX_CHARS = 800


def chunks(text):
    """Режет текст по предложениям на куски не длиннее MAX_CHARS."""
    sentences = re.split(r"(?<=[.!?…])\s+", text.replace("\n", " "))
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > MAX_CHARS:
            yield buf
            buf = ""
        buf = f"{buf} {s}".strip()
    if buf:
        yield buf


def main(src, dst, speaker):
    with open(src, encoding="utf-8") as f:
        text = f.read()
    model, _ = torch.hub.load(
        "snakers4/silero-models", "silero_tts", language="ru", speaker="v4_ru", trust_repo=True
    )
    model.to("cpu")
    pause = torch.zeros(int(SAMPLE_RATE * 0.3))
    parts = list(chunks(text))
    audio = []
    for i, part in enumerate(parts, 1):
        audio += [model.apply_tts(text=part, speaker=speaker, sample_rate=SAMPLE_RATE,
                                  put_accent=True, put_yo=True), pause]
        print(f"{i}/{len(parts)}", flush=True)
    pcm = (torch.cat(audio).clamp(-1, 1) * 32767).short().numpy().tobytes()
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "out.wav")
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "128k", dst], check=True)
    print(f"Готово: {dst}")


if len(sys.argv) < 3:
    sys.exit(__doc__)
main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "xenia")
