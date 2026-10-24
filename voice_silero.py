"""Озвучка русского текста нейромоделью Silero v4 (модель работает локально).

Установка (один раз):
    pip install torch numpy omegaconf
    # нужен ffmpeg в PATH для сборки mp3

Запуск:
    python voice_silero.py text.txt podcast.mp3
    python voice_silero.py text.txt podcast.mp3 baya slow 0.7
    python voice_silero.py text.txt out.mp3 xenia,baya,aidar,eugene   # по файлу на голос

Аргументы: голос(а), темп (x-slow, slow, medium, fast), пауза между предложениями в секундах.
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
from xml.sax.saxutils import escape

import torch

SAMPLE_RATE = 48000
RATES = {"x-slow", "slow", "medium", "fast", "x-fast"}


def sentences(text):
    """Предложения; пустая строка между абзацами даёт метку None (длинная пауза)."""
    for para in re.split(r"\n\s*\n", text):
        for s in re.split(r"(?<=[.!?…])\s+", para.replace("\n", " ").strip()):
            if s:
                yield s
        yield None


def synth(model, text, speaker, rate, pause):
    audio = []
    for s in sentences(text):
        if s is None:
            audio.append(torch.zeros(int(SAMPLE_RATE * pause * 1.5)))
            continue
        ssml = f'<speak><prosody rate="{rate}">{escape(s)}</prosody></speak>'
        audio.append(model.apply_tts(ssml_text=ssml, speaker=speaker, sample_rate=SAMPLE_RATE,
                                     put_accent=True, put_yo=True))
        audio.append(torch.zeros(int(SAMPLE_RATE * (pause + 0.25 * s.endswith(("?", "!"))))))
        print(".", end="", flush=True)
    print()
    return (torch.cat(audio).clamp(-1, 1) * 32767).short().numpy().tobytes()


def save_mp3(pcm, dst):
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "out.wav")
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "128k", dst], check=True)


def main(src, dst, speakers, rate, pause):
    if rate not in RATES:
        sys.exit(f"Темп должен быть один из: {', '.join(sorted(RATES))}")
    with open(src, encoding="utf-8") as f:
        text = f.read()
    model, _ = torch.hub.load(
        "snakers4/silero-models", "silero_tts", language="ru", speaker="v4_ru", trust_repo=True
    )
    model.to("cpu")
    names = speakers.split(",")
    for name in names:
        out = dst if len(names) == 1 else f"{os.path.splitext(dst)[0]}_{name}.mp3"
        save_mp3(synth(model, text, name, rate, pause), out)
        print(f"Готово: {out}")


if len(sys.argv) < 3:
    sys.exit(__doc__)
a = sys.argv
main(a[1], a[2], a[3] if len(a) > 3 else "xenia", a[4] if len(a) > 4 else "slow", float(a[5]) if len(a) > 5 else 0.7)
