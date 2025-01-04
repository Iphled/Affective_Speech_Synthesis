#Umformung von Audio in Graph
#Eingabe: Audio
#Ausgabe: drei Graphen: Tempo, Lautstärke, Pitch
import math

from pydub import AudioSegment
from scipy.io.wavfile import read
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import numpy as np

def mp3_wav(path):
    sound = AudioSegment.from_mp3(path)
    path=path.replace("mp3","wav")
    sound.export(path, format="wav")
    return path

def audio_to_pitch_over_time(path):
    if path.endswith('.mp3'):
        path=mp3_wav(path)
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

def audio_to_volume_over_time(path,shorten=False):
    if path.endswith('.mp3'):
        path=mp3_wav(path)
        print(path)
    rate, data = read(path)
    length = data.shape[0] / rate
    time = np.linspace(0., length, data.shape[0])
    values = list()
    for i in range(data.shape[0]):
        if (type(data[i]) is list or type(data[i]) is tuple) and len(data[i])> 1:
            if data[i][0] > data[i][1]:
                values.append(data[i][0])
            else:
                values.append(data[i][1])
        elif (type(data[i]) is list or type(data[i]) is tuple) and len(data[i])==1:
            values.append(data[i][0])
        else:
            values.append(data[i])
    if shorten:
        values2=list()
        sectlen = int(len(values) / 1000)
        for i in range(1000):
            mittel = 0
            count = 0
            for j in range(i * sectlen, (i + 1) * sectlen):
                if j < len(values):
                    count = count + 1
                    m = values[j]
                    mittel = mittel + 10 ** ((values[j]) / 10)
            if (mittel == 0):
                values2.append(values[j])
            else:
                values2.append(10 * math.log(mittel / count, 10))
        return values2, length
    else:
        return values, length
