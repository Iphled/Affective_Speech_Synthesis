import audiostretchy.stretch
import librosa
import numpy as np
import pydub
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

    # sound1 6 dB louder
    #louder = sound1 + 6

    # sound1, with sound2 appended (use louder instead of sound1 to append the louder version)
    combined=None
    for part in parts:
        if combined is None:
            combined=part
        else:
            combined=combined+part

    return combined
    # simple export
    #file_handle = combined.export("/path/to/output.mp3", format="mp3")

def stretch(audio,factor):
    return librosa.effects.time_stretch(audio[0], rate=factor)

def change_loudness(audio,start,end):
    length=len(audio)
    for i,p in enumerate(audio):
        factor=start+(end-start)/length*i
        audio[i]=p*factor
    return audio

def write_back_audio(time,volume,pitch,audio,length):
    parts=segment_audio(audio,length/len(time)*1000)
    for i, chunk in enumerate(parts):
        chunk_name = '{0}.wav'.format(i)
        chunk_name2 = '{0}s.wav'.format(i)
        print('exporting', chunk_name)
        chunk.export(chunk_name, format='wav')
        samples = chunk.get_array_of_samples()
        new_sound = chunk._spawn(samples)
        arr = np.array(samples).astype(np.float32)
        print(type(arr))
        # print(arr)
        # then modify samples...
        y, index = librosa.effects.trim(arr)
        y,sr=librosa.load(chunk_name)
        y= stretch((y,sr), time[i])
        y = np.array(y * (1 << 15), dtype=np.int16)
        if i==0:
            y=change_loudness(y,volume[i],volume[i])
        elif i==len(parts)-1:
            y=change_loudness(y,volume[i-1],volume[i-1])
        else:
            y=change_loudness(y,volume[i-1],volume[i])
        parts[i] = pydub.AudioSegment(
            y.tobytes(),
            frame_rate=sr,
            sample_width=y.dtype.itemsize,
            channels=1
        )
        parts[i].export(chunk_name2, format='wav')


    audio=combine_audio(parts)

    return audio

rate, data = read("data/test.wav")
audio = AudioSegment.from_file('data/test.wav', 'r')

length = data.shape[0] / rate
final=write_back_audio([1.2,0.8,1.2],[3,0.3],[2,1],audio,length)
final.export("final.wav", format="wav")