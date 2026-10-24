"""Озвучка текстового файла бесплатными нейроголосами Microsoft Edge (edge-tts).

Запуск:
    pip install edge-tts
    python voice.py text.txt podcast.mp3
    python voice.py text.txt podcast.mp3 ru-RU-DmitryNeural

Голоса: ru-RU-SvetlanaNeural (женский), ru-RU-DmitryNeural (мужской).
Текст отправляется на сервер Microsoft, не озвучивайте конфиденциальное.
"""
import asyncio
import sys

import edge_tts


async def main(src, dst, voice):
    with open(src, encoding="utf-8") as f:
        text = f.read()
    await edge_tts.Communicate(text, voice, rate="-5%").save(dst)
    print(f"Готово: {dst}")


if len(sys.argv) < 3:
    sys.exit(__doc__)
asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "ru-RU-SvetlanaNeural"))
