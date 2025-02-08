import math
import os
import array

import audiostretchy.stretch
import librosa
import numpy
import numpy as np
import pydub
from cytoolz import remove
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io.wavfile import read
import  soundfile
import wave


def segment_audio(audio,parts):
    chunk_sizes = [parts] # pydub calculates in millisec
    chunks=None
    for chunk_length_ms in chunk_sizes:
        chunks = make_chunks(audio,chunk_length_ms) #Make chunks of one sec
    return chunks


def combine_audio(parts):

    combined=None
    for part in parts:
        if combined is None:
            combined=part
        else:
            combined=combined+part

    return combined


def stretch(audio,factor):
    return librosa.effects.time_stretch(audio[0], rate=factor)

def change_loudness(audio,start,end,emphasis,median,shaky):
    length=len(audio)
    vib=0
    if emphasis == 1:
        for i,p in enumerate(audio):
            factor=start+(end-start)/length*i
            audio[i]=p*factor+vib
            if shaky:
                vib=10*math.sin(i*100)
    else:
        for i,p in enumerate(audio):
            factor=(start+(end-start)/length*i)
            if p>0:
                p=max(emphasis*((p*factor)-median)+median,0)
            else :
                p=min(0,emphasis*((p*factor)+median)-median)
            audio[i]=p*factor+vib
            if shaky:
                vib=math.sin(i/100)
    return audio

def change_pitch(audio,steps):
    return librosa.effects.pitch_shift(audio[0], sr=audio[1], n_steps=steps)

def input_pause(audio,median,pause,timeout):
    ar = array.array(audio.array_type, audio._data)
    a2 = list(ar)
    time =0
    stretch=timeout*pause
    for i in range(len(a2)):
        if time==0:
            if abs(a2[i])<median/2:
                time=int(timeout)+timeout*pause
                stretch=timeout*pause
        else:
            time-=1
            stretch=stretch-1
            if stretch>0:
                a2.insert(i+1,a2[i])

    res = array.array(ar.typecode, a2)
    audio = audio._spawn(res)
    return audio

def write_back_audio(time,volume,pitch,audio,length,pause=0,emphasis=1,shaky=False):
    audio_name = 'test.wav'
    audio.export(audio_name, format='wav')

    y, sr = librosa.load(audio_name, dtype=numpy.float64)
    os.remove(audio_name)
    i=0
    not0=0
    yt,index=librosa.effects.trim(y)
    audio = pydub.AudioSegment(
                yt.tobytes(),
                frame_rate=sr,
                sample_width=y.dtype.itemsize,
                channels=1
            )

    median=0
    if pause!=0 or emphasis!=1:
        for i in range(len(y)):
            y[i]=abs(y[i])
        asort=y.copy()
        asort.sort()
        median=asort[len(asort)//2]

    if pause!=0:
        input_pause(audio,median,pause,len(audio)/length*0.1)

    parts = segment_audio(audio, length / len(time) * 1000)


    for i, chunk in enumerate(parts):
        if i<len(pitch):
            chunk_name = '{0}.wav'.format(i)
            chunk.export(chunk_name, format='wav')

            y,sr=librosa.load(chunk_name,dtype=numpy.float64)
            os.remove(chunk_name)
            if i<len(time):
                y= stretch((y,sr), time[i])

            if i==0:
                y=change_loudness(y,volume[i],volume[i],emphasis,median,shaky)
            elif i==len(parts)-1:
                y=change_loudness(y,volume[i-1],volume[i-1],emphasis,median,shaky)
            else:
                if i<len(volume):
                    y=change_loudness(y,volume[i-1],volume[i],emphasis,median,shaky)
            y= change_pitch((y, sr), pitch[i])
            y = np.array(y * (1 << 15), dtype=np.int16)
            parts[i] = pydub.AudioSegment(
                y.tobytes(),
                frame_rate=sr,
                sample_width=y.dtype.itemsize,
                channels=1
            )

    audio=combine_audio(parts)
    return audio

rate, data = read("data/test.wav")
audio = AudioSegment.from_file('data/test.wav', 'r')

length = data.shape[0] / rate
final=write_back_audio([1.2,0.8,1.2],[3,0.3],[2,1,6],audio,length)
final.export("final.wav", format="wav")