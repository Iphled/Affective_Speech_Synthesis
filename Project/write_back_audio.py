import os

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

def change_loudness(audio,start,end):
    length=len(audio)
    for i,p in enumerate(audio):
        factor=start+(end-start)/length*i
        audio[i]=p*factor
    return audio

def change_pitch(audio,steps):
    return librosa.effects.pitch_shift(audio[0], sr=audio[1], n_steps=steps)

def write_back_audio(time,volume,pitch,audio,length):
    parts=segment_audio(audio,length/len(time)*1000)
    for i, chunk in enumerate(parts):
        if i<len(pitch):
            chunk_name = '{0}.wav'.format(i)
            chunk.export(chunk_name, format='wav')

            y,sr=librosa.load(chunk_name,dtype=numpy.float64)
            os.remove(chunk_name)
            if i<len(time):
                y= stretch((y,sr), time[i])

            if i==0:
                y=change_loudness(y,volume[i],volume[i])
            elif i==len(parts)-1:
                y=change_loudness(y,volume[i-1],volume[i-1])
            else:
                if i<len(volume):
                    y=change_loudness(y,volume[i-1],volume[i])
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