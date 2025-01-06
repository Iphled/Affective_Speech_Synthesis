"""
Segment audio into word-level chunks.
input:  audio .wav
Output: audio .wav, start, end, word for each chunk
"""
from vosk import Model, KaldiRecognizer
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pydub import AudioSegment
import soundfile as sf
import librosa
import librosa.display
from numba import jit
import numpy as np
import wave
import json
import time
import os


def transcribe_with_vosk(audio_path, model_path='speech_to_text_model\\vosk-model-en-us-0.22'):
    """
    Transcribe audio and retrieve word-level timestamps using Vosk.

    :param audio_path: Path to the input audio file (WAV format).
    :param model_path: Path to the Vosk model directory.
    :return list: List of dictionaries with word and its start and end times.
    """
    # load Vosk model and audio
    model = Model(model_path)
    wf = wave.open(audio_path, "rb")

    # check if audio format is valid
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000]:
        convert_audio_to_mono_pcm(input_path=audio_path, output_path=audio_path)
        time.sleep(0.01)
        wf = wave.open(audio_path, 'rb')

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)  # enable word-level timestamps

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec_res = json.loads(rec.PartialResult())
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            results.extend(res.get("result", []))

    final = json.loads(rec.FinalResult())
    wf.close()
    return final['result']


def get_local_peaks(signal, distance, window_size=3):
    peaks, _ = find_peaks(signal, distance=distance)
    return peaks


@jit(nopython=True)
def lowest_local(arr, index, range_size):
    n = len(arr)

    # define time range
    start = max(0, index - range_size)
    end = min(n, index + range_size + 1)

    arr_range = arr[start:end]
    lowest_idx = np.argmin(arr_range) + start

    return lowest_idx


def find_closest_number(arr, target):
    # calculate difference
    differences = np.abs(arr - target)
    # smallest difference -> the closest number
    closest_index = np.argmin(differences)
    return arr[closest_index], closest_index


def adjust_word_timestamps(word_timestamps, audio, sr):
    window_size = 3
    new_word_timestamps = []
    for i, word_data in enumerate(word_timestamps):
        new_word_timestamps.append(word_data)

    def relu(x):
        return np.maximum(0, x)

    for i, word_data in enumerate(new_word_timestamps):
        start_sample = int(word_data["start"] * sr)
        end_sample = int(word_data["end"] * sr)
        for tag, idx_in_audio in zip(['start', 'end'], [start_sample, end_sample]):
            audio_peaks = get_local_peaks(relu(audio), distance=50)
            audio_over_curve = np.convolve(audio[audio_peaks], np.ones(window_size), 'same') / window_size

            _, idx_in_peaks = find_closest_number(audio_peaks, target=idx_in_audio)
            closest_minimum_audio = lowest_local(audio_over_curve, index=idx_in_peaks, range_size=10)

            # plt.plot(audio)
            # plt.plot([idx_in_audio], [audio[idx_in_audio]])
            # plt.plot(audio_peaks, audio[audio_peaks], 'x')
            # plt.plot([audio_peaks[closest_minimum_audio]], [audio[audio_peaks[closest_minimum_audio]]], 'v')
            # plt.plot([idx_in_audio], [audio[idx_in_audio]], 'o')
            # plt.show(block=True)

            # check start/end for near silence
            # if audio[audio_peaks[closest_minimum_audio]] / audio.max() < 0.05:
            new_word_timestamps[i][tag] = audio_peaks[closest_minimum_audio] / sr
    return new_word_timestamps


def segment_audio_by_words(audio_path, model_path, output_dir=None):
    """
    Segment audio into word-level chunks using Vosk word-level timestamps.

    :param audio_path: Path to the input audio file.
    :param model_path: Path to the Vosk model directory.
    :param output_dir: Directory to save the word-level audio chunks.
    """
    # load audio
    audio, sr = librosa.load(audio_path, sr=None)

    # get word start and end timestamps
    word_timestamps = transcribe_with_vosk(audio_path, model_path)

    # adjust word timestamp by using the closest local minimum of all peaks
    adj_word_timestamps = adjust_word_timestamps(word_timestamps, audio, sr)

    plot_audio_with_word_segments(audio_path, word_timestamps)
    plot_audio_with_word_segments(audio_path, adj_word_timestamps)

    # segment audio and save each word with its corresponding .wav as a separate file
    for i, word_data in enumerate(word_timestamps):
        start_sample = int(word_data["start"] * sr)
        end_sample = int(word_data["end"] * sr)
        word_audio = audio[start_sample:end_sample]
        word_timestamps[i]['audio'] = word_audio

        # save each word segment (optional)
        if output_dir is not None:
            output_path = f"{output_dir}\\word_{i + 1}_{word_data['word']}.wav"
            sf.write(output_path, word_audio, sr)
            print(f"Saved: {output_path}")
    return word_timestamps


def audio_transcript(audio_path, model_path, output_dir='word_segments'):
    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # segment the audio
    word_timestamp_audio = segment_audio_by_words(audio_path, model_path, output_dir)
    return word_timestamp_audio


def convert_audio_to_mono_pcm(input_path, output_path, sample_rate=16000):
    """
    Convert audio to WAV format, mono PCM, with specified sample rate.
    (necessary for vosk transcription)

    :param input_path: Path to the input audio file.
    :param output_path: Path to save the converted audio file.
    :param sample_rate: Target sample rate (e.g., 8000 or 16000 Hz).
    """
    # load the audio
    audio = AudioSegment.from_file(input_path)

    # convert to mono and set sample rate
    audio = audio.set_channels(1).set_frame_rate(sample_rate)

    # export the audio in WAV format
    audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    print(f"Converted audio saved to: {output_path}")


def plot_audio_with_word_segments(audio_path, word_timestamps, output_plot=None):
    # load the audio
    audio, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)

    # generate time axis for the audio waveform
    time = np.linspace(0, duration, len(audio))

    # plot the audio waveform
    plt.figure(figsize=(12, 6))
    librosa.display.waveshow(audio, sr=sr, alpha=0.5)
    plt.title("Audio Signal with Word Segments")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # overlay word segments
    for word_data in word_timestamps:
        start_time = word_data["start"]
        end_time = word_data["end"]
        word = word_data["word"]

        # highlight word segment
        plt.axvspan(start_time, end_time, color='yellow', alpha=0.3, label=f"{word} [{start_time:.2f}-{end_time:.2f}s]")
        # annotate word
        plt.text((start_time + end_time) / 2, 0.6, word, fontsize=10, ha='center', color='blue', rotation=45)

    # show the plot (and optionally save it)
    plt.tight_layout()
    plt.legend(loc="upper right")
    if output_plot:
        plt.savefig(output_plot)
    plt.show(block=True)


def segment_audio_by_silence(audio, sr, top_db=20, frame_length=1024, hop_length=512):
    """
    Segment audio into chunks based on silence detection

    :param audio: Audio signal.
    :param sr: Sampling rate.
    :param top_db: Threshold in decibels for silence.
    :param frame_length: Frame length for analysis.
    :param hop_length: Hop length for analysis.
    :return list of tuples (start_sample, end_sample) for each detected segment.
    """
    # detect non-silent intervals
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    return intervals


if __name__ == '__main__':
    # used example text for audio:
    # if you thought I lived in new york why in the world didn't you come and see me, the lady inquired.
    model_path = 'speech_to_text_model\\vosk-model-en-us-0.22'  # path to Vosk model
    audio_path1 = 'fmttest_n_1.wav'
    audio_path2 = 'test_Angry_1.wav'
    word_timestamp_audio1 = audio_transcript(audio_path1, model_path, output_dir='neutral_word_segment')
    word_timestamp_audio2 = audio_transcript(audio_path2, model_path, output_dir='emotional_word_segment')
