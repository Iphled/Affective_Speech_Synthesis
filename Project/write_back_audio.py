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

def write_back_audio(time,volume,pitch,audio):
    return audio