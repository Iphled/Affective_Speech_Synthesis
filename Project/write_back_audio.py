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


def segment_audio(audio,num):
    parts=len(audio[0])//num

    chunks = [audio[0][i:i + parts] for i in range(0, len(audio[0]), parts)][:num]
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
    if emphasis <1:
        start=start/10
        end=end/10
    if emphasis>1:
        start = start * 2
        end = end * 2
    length=len(audio)
    vib=0
    for i,p in enumerate(audio):
        factor=start+(end-start)/length*i
        audio[i]=p*factor+vib
    if emphasis!=1:
        if emphasis<1:
            audio=librosa.effects.deemphasis(audio)
        elif emphasis>1:
            audio=librosa.effects.preemphasis(audio)
            audio = librosa.effects.preemphasis(audio)

    return audio


def change_pitch(audio,steps):
    return librosa.effects.pitch_shift(audio[0], sr=audio[1], n_steps=steps)

def input_pause(y,sr,median,pause):
    second = []
    for s in range(0, len(y), sr):
        second.append(np.abs(y[s:s + sr]).mean())
    second.sort()
    median=second[len(second)//2]
    segments=librosa.effects.split(y,top_db=10, ref=np.max)
    segm=[]
    max=0
    for segment in segments:
        if segment[1]>max:
            segm.append(y[max:segment[1]])
            max=segment[1]
    if max<len(y)-1:
        segm.append(y[max:])
    p=np.array([0]*int(sr*pause))
    for i in range(len(segm)-1):
        segm[i]=np.concatenate((segm[i],p),axis=0)
    y=np.concatenate(segm,axis=0)
    return y
    #ar = array.array(audio.array_type, audio._data)
    #a2 = list(ar)
    #time =0
    #stretch=timeout*pause
    #for i in range(len(a2)):
    #    if time==0:
    #        if abs(a2[i])<median/2:
    #            time=int(timeout)+timeout*pause
    #            stretch=timeout*pause
    #    else:
    #        time-=1
    #        stretch=stretch-1
    #        if stretch>0:
    #            a2.insert(i+1,a2[i])

    #res = array.array(ar.typecode, a2)
    #audio = audio._spawn(res)
    #return audio

def write_back_audio(time,volume,pitch,audio,length,pause=0,emphasis=1,shaky=False):
    audio_name = 'test.wav'
    audio.export(audio_name, format='wav')

    y, sr = librosa.load(audio_name, dtype=numpy.float64)
    os.remove(audio_name)
    i=0
    not0=0
    y,index=librosa.effects.trim(y,top_db=50,ref=np.max)

    median=0
    y2=y.copy()
    if pause!=0 or emphasis!=1:
        for i in range(len(y)):
            y2[i]=abs(y2[i])
        y2.sort()
        median=y2[len(y2)//2]

    if pause!=0:
        y=input_pause(y,sr,median,pause)

    parts = segment_audio((y,sr), len(time))

    if shaky:
        noise = np.random.normal(0, y.std(), y.size)
        y = y + noise * 0.3
        for i,p in enumerate(y):
            ad=2*math.sin(i/100)
            y[i]=p+ad


    for i, chunk in enumerate(parts):
        if i<len(pitch):

            y=chunk
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

#rate, data = read("data/1002_IEO_NEU_XX.wav")
#audio = AudioSegment.from_file('data/1002_IEO_NEU_XX.wav', 'r')

#length = data.shape[0] / rate
#final=write_back_audio([1.2,0.8,1.2],[3,0.5],[2,1,6],audio,length,pause=1,shaky=True)
#final.export("final.wav", format="wav")

def writeback_sad(audio,length):
    return write_back_audio([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4], [2, 2, 2, 1.5, 1, 0.7, 0.2],
                             [0.5, 0.5, 0.5, 0.1, -1, -2, -2, -2], audio, length, emphasis=0.5)

def writeback_happy(audio,length):
    return write_back_audio([1,1,1,1.1,1.1,1.2,1.2,1.2],[1,1,1,1.7,2,2,1.5],[2,2,0,0.3,1,4,6,6],audio,length)

def writeback_angry(audio,length):
    return write_back_audio([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], [5, 5, 5, 4, 3, 3, 2], [-1, 0, 0, 0.3, 0.7, 1, 2, 3],
                     audio, length, emphasis=1.4, pause=1)

def writeback_fearful(audio,length):
    final = write_back_audio([1, 1, 1, 1, 0.8, 0.8, 1, 1], [5, 5, 5, 2, 5, 5, 5], [2, 2, 2, 2, 2, 3, 3, 3], audio,
                             length, emphasis=1.5)