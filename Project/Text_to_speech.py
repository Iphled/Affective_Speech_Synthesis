# Enthält das nicht-emotionale TTS
# Eingabe: Ein Satz
# Ausgabe: Das dazugehörige Audio
from gtts import gTTS


def text_to_speech(text):
    audio = gTTS(text=text, lang='en')
    return audio
