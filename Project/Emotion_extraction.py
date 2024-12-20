# Enthält die Extraktion der Emotion
# Eingabe: Ein Satz
# Ausgabe: Eine Emotion von

from transformers import pipeline
import pandas as pd

def extract_from_text(text):
    model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    df=pd.DataFrame.from_dict({'text':[text]})
    emotion=model(df["text"].values.tolist())
    if emotion!="surprise":
        return emotion[0]["label"]
    else:
        return "neutral"



def index_from_emotion(emotion, emotions):
    for i in range(len(emotions)):
        if emotions[i].lower() == emotion.lower():
            return i
    return -1
