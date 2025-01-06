# Umformung von Audio in Graph
# Eingabe: Audio
# Ausgabe: drei Graphen: Tempo, Lautstärke, Pitch
from pydub import AudioSegment
from scipy.io.wavfile import read
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import numpy as np


def mp3_wav(path):
    sound = AudioSegment.from_mp3(path)
    path = path.replace("mp3", "wav")
    sound.export(path, format="wav")
    return path


def audio_to_pitch_over_time(path):
    if path.endswith('.mp3'):
        path = mp3_wav(path)
        print(path)
    y, sr = librosa.load(path)
    chromagram_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    l = list()
    for i in range(chromagram_stft.shape[1]):
        max = 0
        line = -1
        for j in range(chromagram_stft.shape[0]):
            if chromagram_stft[j][i] > max:
                max = chromagram_stft[j][i]
                line = j
        l.append(line)
    return l


def audio_to_volume_over_time(path):
    if path.endswith('.mp3'):
        path = mp3_wav(path)
        print(path)
    rate, data = read(path)
    length = data.shape[0] / rate
    time = np.linspace(0., length, data.shape[0])
    values = list()
    for i in range(data.shape[0]):
        if data[i][0] > data[i][1]:
            values.append(data[i][0])
        else:
            values.append(data[i][1])

    return values, time
