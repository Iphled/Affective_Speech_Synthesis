"""
Segment audio into word-level chunks.
input:  audio .wav
Output: audio .wav, start, end, word for each chunk
"""
from vosk import Model, KaldiRecognizer
from scipy.signal import find_peaks
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pydub import AudioSegment
import soundfile as sf
import librosa
import librosa.display
from numba import jit
import numpy as np
import torchaudio
import torchaudio.functional as F
import IPython
import torch
import wave
import json
import time
import os


def transcribe_with_vosk(audio_path, model_path='speech_to_text_model\\vosk-model-en-us-0.22'):
    """
    Transcribe audio and retrieve word-level timestamps using Vosk.

    :param audio_path: lath to the input audio file (WAV format).
    :param model_path: lath to the Vosk model directory.
    :return word_timestamps: list of dictionaries with word and its start and end times.
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


def get_local_peaks(signal, distance):
    """get peaks of the audio signal

    :param signal: audio signal
    :param distance: minimal distance between neighboring signal peaks
    :return: list of peaks
    """
    peaks, _ = find_peaks(signal, distance=distance)
    return peaks


@jit(nopython=True)
def lowest_local(arr, index, range_size):
    """get the index of the lowest value in a given range from the given index in the arr

    :param arr: audio signal
    :param index: given index of the audio signal
    :param range_size: maximum distance to the given index for the lowest value to occur
    :return: index of the lowest value in the neighborhood of the given index
    """
    n = len(arr)

    # define time range
    start = max(0, index - range_size)
    end = min(n, index + range_size + 1)

    arr_range = arr[start:end]
    lowest_idx = np.argmin(arr_range) + start

    return lowest_idx


def find_closest_number(arr, target):
    """get the index of the value in the array, which is closest to the target value

    :param arr: list of values
    :param target: some value
    :return tuple: [closest_value_in_array, index_of_closest_value]
    """
    # calculate difference
    differences = np.abs(arr - target)
    # smallest difference -> the closest number
    closest_index = np.argmin(differences)
    return arr[closest_index], closest_index


def adjust_word_timestamps(word_timestamps, audio, sr):
    """timestamps from transcription are sometimes a bit off.
    Fixing the start and end timestamps for the corresponding words.

    :param word_timestamps: list of dictionaries with word and their start and end timestamps
    :param audio: audio signal
    :param sr: sample rate
    :return adjusted_word_timestamps: list of dictionaries with word and their adjusted start and end timestamps
    """
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
    """Segment audio into word-level chunks using Vosk word-level timestamps.

    :param audio_path: Path to the input audio file
    :param model_path: Path to the Vosk model directory
    :param output_dir: Directory to save the word-level audio chunks
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


def audio_transcript(audio_path, model_path, output_dir='neutral_word_segments'):
    """Transcribe audio from path into words and start and end timestamps.

    :param audio_path: path of the audio file (.wav)
    :param model_path: path of the vosk model
    :param output_dir: directory to store the word segmentations
    :return word_timestamps: word segments with their corresponding start and end timestamps
    """
    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # segment the audio
    word_timestamp_audio = segment_audio_by_words(audio_path, model_path, output_dir)
    return word_timestamp_audio


def convert_audio_to_mono_pcm(input_path, output_path, sample_rate=16000):
    """Convert audio to WAV format, mono PCM, with specified sample rate.

    :param input_path: path to the input audio file.
    :param output_path: path to save the converted audio file.
    :param sample_rate: target sample rate (e.g., 8000 or 16000 Hz).
    """
    # load the audio
    audio = AudioSegment.from_file(input_path)

    # convert to mono and set sample rate
    audio = audio.set_channels(1).set_frame_rate(sample_rate)

    # export the audio in WAV format
    audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    print(f"Converted audio saved to: {output_path}")


def plot_audio_with_word_segments(audio_path, word_timestamps, output_plot=None):
    """plot the segmented audio with overlaying start and end timestamps

    :param audio_path: path to the input audio file.
    :param word_timestamps: list of word segments with their corresponding start and end timestamps.
    :param output_plot: path where to store the output plot.
    """
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
        start_time = int(word_data["start"] * sr)
        end_time = int(word_data["end"] * sr)
        word_data['audio'] = audio[start_time:end_time]
        word = word_data["word"]
        plt.plot(np.arange(start_time, end_time), word_data['audio'])

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


def plot_frame_wise_class_probabilities(emission):
    fig, ax = plt.subplots()
    img = ax.imshow(emission.T)
    ax.set_title("Frame-wise class probability")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
    fig.tight_layout()

    plt.show(block=True)


def plot_trellis(trellis):
    fig, ax = plt.subplots()
    img = ax.imshow(trellis.T, origin="lower")
    ax.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
    ax.annotate("+ Inf", (trellis.size(0) - trellis.size(1) / 5, trellis.size(1) / 3))
    fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
    fig.tight_layout()
    plt.show(block=True)


def plot_trellis_with_path(trellis, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    plt.imshow(trellis_with_path.T, origin="lower")
    plt.title("The path found by backtracking")
    plt.tight_layout()
    plt.show(block=True)


def plot_trellis_with_segments(trellis, segments, transcript, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start : seg.end, i] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
    ax1.set_title("Path, label and probability for each label")
    ax1.imshow(trellis_with_path.T, origin="lower", aspect="auto")

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i - 0.7), size="small")
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 3), size="small")

    ax2.set_title("Label probability with and without repetation")
    xs, hs, ws = [], [], []
    for seg in segments:
        if seg.label != "|":
            xs.append((seg.end + seg.start) / 2 + 0.4)
            hs.append(seg.score)
            ws.append(seg.end - seg.start)
            ax2.annotate(seg.label, (seg.start + 0.8, -0.07))
    ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

    xs, hs = [], []
    for p in path:
        label = transcript[p.token_index]
        if label != "|":
            xs.append(p.time_index + 1)
            hs.append(p.score)

    ax2.bar(xs, hs, width=0.5, alpha=0.5)
    ax2.axhline(0, color="black")
    ax2.grid(True, axis="y")
    ax2.set_ylim(-0.1, 1.1)
    fig.tight_layout()
    plt.show(block=True)


def plot_alignments(trellis, segments, word_segments, waveform, sample_rate):
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start : seg.end, i] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1)

    ax1.imshow(trellis_with_path.T, origin="lower", aspect="auto")
    ax1.set_facecolor("lightgray")
    ax1.set_xticks([])
    ax1.set_yticks([])

    for word in word_segments:
        ax1.axvspan(word.start - 0.5, word.end - 0.5, edgecolor="white", facecolor="none")

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i - 0.7), size="small")
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 3), size="small")

    # The original waveform
    ratio = waveform.size(0) / sample_rate / trellis.size(0)
    ax2.specgram(waveform, Fs=sample_rate)
    for word in word_segments:
        x0 = ratio * word.start
        x1 = ratio * word.end
        ax2.axvspan(x0, x1, facecolor="none", edgecolor="white", hatch="/")
        ax2.annotate(f"{word.score:.2f}", (x0, sample_rate * 0.51), annotation_clip=False)

    for seg in segments:
        if seg.label != "|":
            ax2.annotate(seg.label, (seg.start * ratio, sample_rate * 0.55), annotation_clip=False)
    ax2.set_xlabel("time [second]")
    ax2.set_yticks([])
    fig.tight_layout()
    plt.show(block=True)


def plot_compare_words(audio1, audio2, word):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(range(0, audio1.shape[0]), audio1)
    ax[1].plot(range(0, audio2.shape[0]), audio2)
    plt.title(word)
    plt.show(block=True)
    pass


def segment_audio_by_silence(audio, sr, top_db=20, frame_length=1024, hop_length=512):
    """Segment audio into chunks based on silence detection

    :param audio: audio signal.
    :param sr: sampling rate.
    :param top_db: threshold in decibels for silence.
    :param frame_length: frame length for analysis.
    :param hop_length: hop length for analysis.
    :return list of tuples (start_sample, end_sample) for each detected segment.
    """
    # detect non-silent intervals
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    return intervals


def transcribe_using_whisper():
    import whisper_timestamped as whisper

    audio = whisper.load_audio("fmttest_n_1.wav")

    model = whisper.load_model("small", device="cpu")

    result = whisper.transcribe(model, audio, language="en", naive_approach=True, plot_word_alignment='audio_seg_whisper_ts')

    print(json.dumps(result, indent=2, ensure_ascii=False))
    word_timestamps = result['segments'][0]['words']

    # load the audio
    audio, sr = librosa.load('test_n_1.wav', sr=None)
    # segment audio and save each word with its corresponding .wav as a separate file
    for i, word_data in enumerate(word_timestamps):
        start_sample = int(word_data["start"] * sr)
        end_sample = int(word_data["end"] * sr)
        word_audio = audio[start_sample:end_sample]
        word_timestamps[i]['audio'] = word_audio
        word_timestamps[i]['word'] = word_data['text']
    for word_data in word_timestamps:
        start_time = int(word_data["start"] * sr)
        end_time = int(word_data["end"] * sr)
        word = word_data["word"]
        plt.plot(list(range(start_time, end_time, 1)), word_data['audio'], c='black')

        # highlight word segment
        plt.axvspan(start_time, end_time, color='yellow', alpha=0.3, label=f"{word} [{start_time:.2f}-{end_time:.2f}s]")
        # annotate word
        plt.text((start_time + end_time) / 2, 0.6, word, fontsize=10, ha='center', color='blue', rotation=45)

    # show the plot (and optionally save it)
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.show(block=True)


    return word_timestamps


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1:, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis


def get_word_timestamps_with_wav2vec(audio_path, transcript):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to('cpu')
    labels = bundle.get_labels()
    with torch.inference_mode():
        waveform, _ = torchaudio.load(audio_path)
        emissions, _ = model(waveform.to('cpu'))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()

    print(labels)
    plot_frame_wise_class_probabilities(emission)

    transcript = '|' + transcript.upper().replace(' ', '|').replace(',', '').replace('.', '') + '|'

    dictionary = {c: i for i, c in enumerate(labels)}

    tokens = [dictionary[c] for c in transcript if c in dictionary.keys()]        # char level tokens
    print(list(zip(transcript, tokens)))

    trellis = get_trellis(emission, tokens, blank_id=0)

    plot_trellis(trellis)

    path = backtrack(trellis, emission, tokens)
    for p in path:
        print(p)

    plot_trellis_with_path(trellis, path)

    segments = merge_repeats(path, transcript)
    for seg in segments:
        print(seg)

    plot_trellis_with_segments(trellis, segments, transcript, path)

    word_segments = merge_words(segments)
    for word in word_segments:
        print(word)

    plot_alignments(
        trellis,
        segments,
        word_segments,
        waveform[0],
        bundle.sample_rate
    )
    # output_dir = 'neutral_word_segment'

    for i in range(len(word_segments)):
        word = word_segments[i].label
        audio_word_display, audio_word_np = display_segment(i, waveform, trellis, word_segments, bundle.sample_rate)
        output_path = f"000word_{i + 1}_{word}.wav"
        sf.write(output_path, audio_word_np, bundle.sample_rate)

    print(transcript)
    # IPython.display.Audio(audio_path, autoplay=True)

    pass


def forced_alignment(audio_path, transcript):
    waveform, _ = torchaudio.load(audio_path)
    TRANSCRIPT = transcript.upper().split()

    bundle = torchaudio.pipelines.MMS_FA

    model = bundle.get_model(with_star=False).to('cpu')
    with torch.inference_mode():
        emission, _ = model(waveform.to('cpu'))

    LABELS = bundle.get_labels(star=None)
    DICTIONARY = bundle.get_dict(star=None)
    for k, v in DICTIONARY.items():
        print(f"{k}: {v}")

    tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPT for c in word if c in DICTIONARY.keys()]

    for t in tokenized_transcript:
        print(t, end=" ")
    print()

    aligned_tokens, alignment_scores = align(emission, tokenized_transcript)

    for i, (ali, score) in enumerate(zip(aligned_tokens, alignment_scores)):
        print(f"{i:3d}:\t{ali:2d} [{LABELS[ali]}], {score:.2f}")

    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
    word_spans = unflatten(token_spans, [len(word) for word in TRANSCRIPT])

    num_frames = emission.size(1)
    audio_word_display, audio_word_np = preview_word(waveform, word_spans[0], num_frames, transcript, bundle.sample_rate)

    pass


def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


def preview_word(waveform, spans, num_frames, transcript, sample_rate):
    ratio = waveform.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    print(f"{transcript} ({_score(spans):.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate, autoplay=True), segment.numpy()[0]


def unflatten(list_, lengths):
    #assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device='cpu')
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores


def display_segment(i, waveform, trellis, word_segments, sample_rate):
    ratio = waveform.size(1) / trellis.size(0)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    print(f"{word.label} ({word.score:.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate, autoplay=True), segment.numpy()[0]


def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]


if __name__ == '__main__':
    # used example text for audio:
    transcript_text = "If you thought I lived in new york why in the world didn't you come and see me, the lady inquired."

    # forced_alignment(audio_path='test_n_1.wav', transcript=transcript_text)

    # get_word_timestamps_with_wav2vec(audio_path='test_n_1.wav', transcript=transcript_text)
    # transcribe_using_whisper()
    model_path = 'speech_to_text_model\\vosk-model-en-us-0.22'  # path to Vosk model
    audio_path1 = 'fmttest_n_1.wav'
    audio_path2 = 'test_Angry_1.wav'
    word_timestamp_audio1 = audio_transcript(audio_path1, model_path, output_dir='neutral_word_segment')
    word_timestamp_audio2 = audio_transcript(audio_path2, model_path, output_dir='emotional_word_segment')

    for w_ts1, w_ts2 in zip(word_timestamp_audio1, word_timestamp_audio2):
        audio1 = w_ts1['audio']
        audio2 = w_ts2['audio']
        word = w_ts1['word']
        plot_compare_words(audio1, audio2, word)
