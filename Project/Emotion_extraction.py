# Enthält die Extraktion der Emotion
# Eingabe: Ein Satz
# Ausgabe: Eine Emotion von

def extract_from_text(text):
    return "neutral"


def index_from_emotion(emotion, emotions):
    for i in range(len(emotions)):
        if emotions[i].lower() == emotion.lower():
            return i
    return -1
