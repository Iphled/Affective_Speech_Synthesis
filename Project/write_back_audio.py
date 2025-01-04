import audiostretchy

from pydub import AudioSegment
from pydub.utils import make_chunks

def segment_audio(audio,parts):
    chunk_sizes = [10000] # pydub calculates in millisec
    for chunk_length_ms in chunk_sizes:
        chunks = make_chunks(audio,chunk_length_ms) #Make chunks of one sec
        for i, chunk in enumerate(chunks):
            chunk_name = '{0}.wav'.format(i)
            print ('exporting', chunk_name)
            chunk.export(chunk_name, format='wav')

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
    pass

def write_back_audio(time,volume,pitch,audio,length):
    parts=segment_audio(audio,length/(time)*60*10000)
    for i in range(len(parts)):
        stretch(parts[i],time[i])

    audio=combine_audio(parts)
    return audio