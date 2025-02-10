from pydub import AudioSegment
from scipy.io.wavfile import read
from librosa import feature
import librosa.display
from dtw import dtw
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from librosa.effects import pitch_shift
from librosa.feature import mfcc

from Audio_to_Values import mp3_wav, audio_to_pitch_over_time, audio_to_volume_over_time


def get_local_peaks(signal, distance, window_size=3):
    peaks, _ = find_peaks(signal, distance=distance)
    # plt.plot(signal)
    # plt.plot(peaks, signal[peaks], 'x')
    # plt.show(block=True)

    # plt.plot(over_curve)
    # plt.show(block=True)
    return peaks


def compare_words(audio_w1, audio_w2):
    window_size = 3
    # load two audio signals (or their feature representations)
    audio1, sr1 = librosa.load("neutral_word_segment\\word_3_thought.wav")      # neutral
    audio2, sr2 = librosa.load("emotional_word_segment\\word_3_thought.wav")      # emotional

    audio2 = librosa.effects.time_stretch(audio2, rate=0.7)

    def relu(x):
        return np.maximum(0, x)
    audio1_peaks = get_local_peaks(relu(audio1), distance=50)
    audio2_peaks = get_local_peaks(audio2, distance=50)
    audio1_over_curve = np.convolve(audio1[audio1_peaks], np.ones(window_size), 'same') / window_size
    audio2_over_curve = np.convolve(audio2[audio2_peaks], np.ones(window_size), 'same') / window_size

    # idx_in_audio1 = 6400
    # _, idx_in_peaks = find_closest_number(audio1_peaks, target=idx_in_audio1)
    # closest_minimum_audio1 = np.asarray([lowest_local(audio1_over_curve, index=idx_in_peaks, range_size=10)])
    #
    # plt.plot(audio1)
    # plt.plot(audio1_peaks, audio1[audio1_peaks], 'x')
    # plt.plot(audio1_peaks[closest_minimum_audio1], audio1[closest_minimum_audio1], 'o')
    # plt.show(block=True)

    audio1_over_curve_min = get_local_peaks(1 - audio1_over_curve, distance=100, window_size=3)
    audio2_over_curve_min = get_local_peaks(audio2_over_curve, distance=100, window_size=3)

    audio1_peak_i = audio1_peaks[audio1_over_curve_min]

    plt.plot(audio1)
    plt.plot(audio1_peak_i, audio1[audio1_peak_i], 'x')
    plt.show(block=True)
    # normalize volume
    audio1 = audio1 / audio1.max()
    audio2 = audio2 / audio2.max()

    plt.plot(list(range(audio1.shape[0])), abs(audio1), label='neutral')
    plt.plot(list(range(audio2.shape[0])), abs(audio2), label="emotional")
    plt.legend()
    plt.show(block=True)

    aligned_audio_signal(audio1, audio2)
    # aligned_audio_volume(audio1, audio2)

    # feature extraction (e.g., MFCC)
    mfcc1 = feature.mfcc(y=audio1, sr=sr1, n_mfcc=13)
    mfcc2 = feature.mfcc(y=audio2, sr=sr2, n_mfcc=13)

    # compute DTW alignment using Euclidean distance
    dtw_obj = dtw(mfcc1.T, mfcc2.T, dist_method='euclidean')

    index_1 = dtw_obj.index1
    index_2 = dtw_obj.index2
    distance = dtw_obj.distance
    norm_dist = dtw_obj.normalizedDistance
    path = index_1, index_2

    # align the audio features based on the path
    aligned_mfcc1 = mfcc1[:, index_1]
    aligned_mfcc2 = mfcc2[:, index_2]

    # align the raw audio signals based on the path
    aligned_audio1 = audio1[index_1]
    aligned_audio2 = audio2[index_2]

    # visualize the warping path
    plt.plot(path[0], path[1], 'r')  # Plot alignment path
    plt.title("DTW Alignment Path")
    plt.xlabel("Audio 1 (MFCC indices)")
    plt.ylabel("Audio 2 (MFCC indices)")
    plt.show(block=True)

    # compute pairwise cost matrix
    cost_matrix = cdist(mfcc1.T, mfcc2.T, metric='euclidean')
    print("Cost matrix shape:", cost_matrix.shape)

    # optionally visualize the cost matrix
    plt.imshow(cost_matrix, cmap='viridis', origin='lower')
    plt.title("Cost Matrix")
    plt.xlabel("MFCC 2")
    plt.ylabel("MFCC 1")
    plt.colorbar()
    plt.show(block=True)

    # plot the aligned features
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # plot aligned MFCC for audio 1
    axs[0].imshow(aligned_mfcc1, aspect='auto', origin='lower', cmap='viridis')
    axs[0].set_title("Aligned MFCC - Audio 1")
    axs[0].set_ylabel("MFCC Coefficients")

    # plot aligned MFCC for audio 2
    axs[1].imshow(aligned_mfcc2, aspect='auto', origin='lower', cmap='viridis')
    axs[1].set_title("Aligned MFCC - Audio 2")
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.show(block=True)

    # plot the aligned audio signals in subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # plot aligned audio 1
    axs[0].plot(aligned_audio1, label="Audio 1 (Aligned)", color='blue', alpha=0.7)
    axs[0].set_title("Aligned Audio 1")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    # plot aligned audio 2
    axs[1].plot(aligned_audio2, label="Audio 2 (Aligned)", color='orange', alpha=0.7)
    axs[1].set_title("Aligned Audio 2")
    axs[1].set_xlabel("Aligned Time Steps")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()

    # adjust layout and display
    plt.tight_layout()
    plt.show(block=True)

    print(f"DTW distance: {distance}")


def convert_audio_to_mono_pcm(input_path, output_path, sample_rate=16000):
    """
    Convert audio to WAV format, mono PCM, with specified sample rate.

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


def segment_audio_stft(audio, sr, frame_length=2048, hop_length=512, energy_threshold=0.3):
    """
    Segment audio into words using STFT and spectral analysis.

    :param audio: Audio signal.
    :param sr: Sampling rate.
    :param frame_length: Frame size for STFT.
    :param hop_length: Hop size for STFT.
    :param energy_threshold: Threshold for detecting significant energy changes.
    :return list: A list of tuples (start_sample, end_sample) for each detected segment.
    """
    # compute the STFT of the audio signal
    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(stft)

    # compute the energy of each frame
    energy = np.sum(magnitude ** 2, axis=0)
    energy = energy / np.max(energy)  # Normalize the energy

    # detect peaks in energy to find word boundaries
    peaks, _ = find_peaks(energy, height=energy_threshold, distance=10)

    # determine segment boundaries
    boundaries = np.concatenate(([0], peaks * hop_length, [len(audio)]))
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    return segments, energy


def seg_audio_stft():
    # Load audio file
    audio1, sr1 = librosa.load("test_n_1.wav")
    audio2, sr2 = librosa.load("test_Angry_1.wav")

    # Segment the audio
    segments, energy = segment_audio_stft(audio1, sr1, energy_threshold=0.3)

    # Extract each segment
    chunks = [audio1[int(start):int(end)] for start, end in segments]

    # Plot the audio waveform and energy
    plt.figure(figsize=(14, 8))

    # Plot the waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio1, sr=sr1, alpha=0.6, label="Audio Signal")
    for start, end in segments:
        plt.axvspan(start / sr1, end / sr1, color='red', alpha=0.3, label="Segment" if start == segments[0][0] else "")
    plt.title("Audio Waveform with Detected Word Segments")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Plot the energy
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, len(audio1) / sr1, len(energy)), energy, label="Spectral Energy")
    plt.axhline(y=0.3, color='red', linestyle='--', label="Threshold")
    plt.title("Spectral Energy Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Energy")
    plt.legend()

    plt.tight_layout()
    plt.show(block=True)

    # Print number of segments detected
    print(f"Number of segments detected: {len(chunks)}")


def segment_audio_into_words():
    # Load audio files
    audio1, sr1 = librosa.load("test_n_1.wav")
    audio1, sr1 = librosa.load("test_Angry_1.wav")
    audio2, sr2 = librosa.load("test_Angry_2.wav")

    # Segment both audios
    segments1 = segment_audio_by_silence(audio1, sr1)
    segments2 = segment_audio_by_silence(audio2, sr2)

    # Extract each segment as a separate chunk
    chunks1 = [audio1[start:end] for start, end in segments1]
    chunks2 = [audio2[start:end] for start, end in segments2]

    # Plot the segmented audio
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot audio 1 with segments
    axs[0].plot(audio1, label="Audio 1")
    for start, end in segments1:
        axs[0].axvspan(start, end, color='red', alpha=0.3, label="Segment" if start == segments1[0][0] else "")
    axs[0].set_title("Audio 1 with Detected Word Segments")
    axs[0].legend()

    # Plot audio 2 with segments
    axs[1].plot(audio2, label="Audio 2")
    for start, end in segments2:
        axs[1].axvspan(start, end, color='red', alpha=0.3, label="Segment" if start == segments2[0][0] else "")
    axs[1].set_title("Audio 2 with Detected Word Segments")
    axs[1].legend()

    plt.tight_layout()
    plt.show(block=True)

    # Print the number of chunks
    print(f"Audio 1: {len(chunks1)} segments detected")
    print(f"Audio 2: {len(chunks2)} segments detected")


def test_audio_map():
    # Load two audio signals (or their feature representations)
    audio1, sr1 = librosa.load("test_Angry_1.wav")
    audio2, sr2 = librosa.load("test_n_1.wav")

    # aligned_audio_signal(audio1, audio2)
    # aligned_audio_volume(audio1, audio2)

    # feature extraction (e.g., MFCC)
    mfcc1 = feature.mfcc(y=audio1, sr=sr1, n_mfcc=13)
    mfcc2 = feature.mfcc(y=audio2, sr=sr2, n_mfcc=13)

    # compute DTW alignment using Euclidean distance
    dtw_obj = dtw(mfcc1.T, mfcc2.T, dist_method='euclidean')

    index_1 = dtw_obj.index1
    index_2 = dtw_obj.index2
    distance = dtw_obj.distance
    norm_dist = dtw_obj.normalizedDistance
    path = index_1, index_2

    # align the audio features based on the path
    aligned_mfcc1 = mfcc1[:, index_1]
    aligned_mfcc2 = mfcc2[:, index_2]

    # align the raw audio signals based on the path
    aligned_audio1 = audio1[index_1]
    aligned_audio2 = audio2[index_2]

    # visualize the warping path
    plt.plot(path[0], path[1], 'r')  # Plot alignment path
    plt.title("DTW Alignment Path")
    plt.xlabel("Audio 1 (MFCC indices)")
    plt.ylabel("Audio 2 (MFCC indices)")
    plt.show(block=True)

    # compute pairwise cost matrix
    cost_matrix = cdist(mfcc1.T, mfcc2.T, metric='euclidean')
    print("Cost matrix shape:", cost_matrix.shape)

    # optionally visualize the cost matrix
    plt.imshow(cost_matrix, cmap='viridis', origin='lower')
    plt.title("Cost Matrix")
    plt.xlabel("MFCC 2")
    plt.ylabel("MFCC 1")
    plt.colorbar()
    plt.show(block=True)

    # plot the aligned features
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # plot aligned MFCC for audio 1
    axs[0].imshow(aligned_mfcc1, aspect='auto', origin='lower', cmap='viridis')
    axs[0].set_title("Aligned MFCC - Audio 1")
    axs[0].set_ylabel("MFCC Coefficients")

    # plot aligned MFCC for audio 2
    axs[1].imshow(aligned_mfcc2, aspect='auto', origin='lower', cmap='viridis')
    axs[1].set_title("Aligned MFCC - Audio 2")
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.show(block=True)

    # plot the aligned audio signals in subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # plot aligned audio 1
    axs[0].plot(aligned_audio1, label="Audio 1 (Aligned)", color='blue', alpha=0.7)
    axs[0].set_title("Aligned Audio 1")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    # plot aligned audio 2
    axs[1].plot(aligned_audio2, label="Audio 2 (Aligned)", color='orange', alpha=0.7)
    axs[1].set_title("Aligned Audio 2")
    axs[1].set_xlabel("Aligned Time Steps")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()

    # adjust layout and display
    plt.tight_layout()
    plt.show(block=True)

    print(f"DTW distance: {distance}")


def aligned_audio_volume(audio1, audio2):
    # compute volume (RMS - Root Mean Square) for both audio signals
    frame_length = 2048
    hop_length = 512
    rms1 = librosa.feature.rms(y=audio1, frame_length=frame_length, hop_length=hop_length).flatten()
    rms2 = librosa.feature.rms(y=audio2, frame_length=frame_length, hop_length=hop_length).flatten()

    # perform DTW alignment using Euclidean distance
    alignment = dtw(rms1.reshape(-1, 1), rms2.reshape(-1, 1), dist_method='euclidean')

    # extract the alignment path
    path1, path2 = alignment.index1, alignment.index2

    # align the RMS values based on the path
    aligned_rms1 = rms1[path1]
    aligned_rms2 = rms2[path2]

    # plot the aligned volumes over time
    plt.figure(figsize=(10, 6))

    # plot aligned RMS for audio 1
    plt.plot(aligned_rms1, label="Audio 1 Volume (Aligned)", alpha=0.8)

    # plot aligned RMS for audio 2
    plt.plot(aligned_rms2, label="Audio 2 Volume (Aligned)", alpha=0.8)

    # add labels, legend, and title
    plt.title("Aligned Volume Over Time")
    plt.xlabel("Aligned Time Steps")
    plt.ylabel("Volume (RMS)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)


def aligned_audio_signal(audio1, audio2):
    # compute DTW alignment on the raw audio signals
    alignment = dtw(audio1, audio2, dist_method='euclidean')

    # extract the alignment path
    path1, path2 = alignment.index1, alignment.index2

    # align the original audio signals based on the path
    aligned_audio1 = audio1[path1]
    aligned_audio2 = audio2[path2]

    # plot the aligned original audio signals
    plt.figure(figsize=(12, 6))

    # plot aligned audio 1
    plt.plot(aligned_audio1, label="Audio 1 (Aligned)", alpha=0.8)
    # plot aligned audio 2
    plt.plot(aligned_audio2, label="Audio 2 (Aligned)", alpha=0.8)

    # add labels, legend, and title
    plt.title("Aligned Original Audio Signals")
    plt.xlabel("Aligned Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)


def align_with_dtw(x_1, x_2, fs):
    # Trim silence from the start and end
    x_1, _ = librosa.effects.trim(x_1, top_db=20)
    x_2, _ = librosa.effects.trim(x_2, top_db=20)

    # ___ extract features from chroma representaion ___
    hop_length = 128

    x_1_chroma = librosa.feature.chroma_cqt(y=x_1, sr=fs,
                                            hop_length=hop_length)
    x_2_chroma = librosa.feature.chroma_cqt(y=x_2, sr=fs,
                                            hop_length=hop_length)

    fig, ax = plt.subplots(nrows=2, sharey=True)
    img = librosa.display.specshow(x_1_chroma, x_axis='time',
                                   y_axis='chroma',
                                   hop_length=hop_length, ax=ax[0])
    ax[0].set(title='Chroma Representation of $X_1$')
    librosa.display.specshow(x_2_chroma, x_axis='time',
                             y_axis='chroma',
                             hop_length=hop_length, ax=ax[1])
    ax[1].set(title='Chroma Representation of $X_2$')
    fig.colorbar(img, ax=ax)
    plt.show(block=True)

    # ___ align chroma representations ___
    D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, metric='cosine')
    wp_s = librosa.frames_to_time(wp, sr=fs, hop_length=hop_length)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis='time', y_axis='time', sr=fs,
                                   cmap='gray_r', hop_length=hop_length, ax=ax)
    ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
    ax.set(title='Warping Path on Acc. Cost Matrix $D$',
           xlabel='Time $(X_2)$', ylabel='Time $(X_1)$')
    fig.colorbar(img, ax=ax)
    plt.show(block=True)

    # ___ visualize time maatchings ___
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8, 4))

    # Plot x_2
    librosa.display.waveshow(x_2, sr=fs, ax=ax2)
    ax2.set(title='Faster Version $X_2$')

    # Plot x_1
    librosa.display.waveshow(x_1, sr=fs, ax=ax1)
    ax1.set(title='Slower Version $X_1$')
    ax1.label_outer()

    n_arrows = 20
    for tp1, tp2 in wp_s[::len(wp_s) // n_arrows]:
        # Create a connection patch between the aligned time points
        # in each subplot
        con = ConnectionPatch(xyA=(tp1, 0), xyB=(tp2, 0),
                              axesA=ax1, axesB=ax2,
                              coordsA='data', coordsB='data',
                              color='r', linestyle='--',
                              alpha=0.5)
        con.set_in_layout(False)  # This is needed to preserve layout
        ax2.add_artist(con)
    plt.show(block=True)

    return x_1, x_2


def spectrogram_difference():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR = ROOT_DIR + '\\audio_files\\AudioWAV'
    # Load neutral and emotional audio
    neutral_audio, sr = librosa.load(AUDIO_DIR + "\\1001_DFA_NEU_XX.wav", sr=None)
    emotional_audio, _ = librosa.load(AUDIO_DIR + "\\1001_DFA_ANG_XX.wav", sr=sr)  # Ensure same sample rate

    aligned_audio = align_with_dtw(neutral_audio, emotional_audio, sr)

    # Compute spectrograms
    n_fft = 1024
    hop_length = 512

    S_neutral = np.abs(librosa.stft(neutral_audio, n_fft=n_fft, hop_length=hop_length))
    S_emotional = np.abs(librosa.stft(aligned_emotional_audio, n_fft=n_fft, hop_length=hop_length))

    # Convert to log scale
    log_S_neutral = librosa.amplitude_to_db(S_neutral)
    log_S_emotional = librosa.amplitude_to_db(S_emotional)

    # Compute spectrogram difference
    spectrogram_diff = log_S_emotional - log_S_neutral

    # Extract pitch (F0) differences
    f0_neutral, voiced_flag_neutral, _ = librosa.pyin(neutral_audio, fmin=50, fmax=300)
    f0_emotional, voiced_flag_emotional, _ = librosa.pyin(emotional_audio, fmin=50, fmax=300)

    f0_difference = f0_emotional - f0_neutral

    # Extract MFCC differences
    mfcc_neutral = mfcc(y=neutral_audio, sr=sr, n_mfcc=13)
    mfcc_emotional = mfcc(y=emotional_audio, sr=sr, n_mfcc=13)

    mfcc_difference = mfcc_emotional - mfcc_neutral

    # Apply differences to neutral audio
    S_modified = S_neutral * (10 ** (spectrogram_diff / 20))  # Convert dB back to magnitude

    # Reconstruct modified audio
    modified_audio = librosa.griffinlim(S_modified, hop_length=hop_length)

    # Save output
    librosa.output.write_wav("modified_emotional.wav", modified_audio, sr=sr)

    # Visualization
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    librosa.display.specshow(log_S_neutral, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.title("Neutral Spectrogram")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    librosa.display.specshow(log_S_emotional, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.title("Emotional Spectrogram")
    plt.colorbar()
    plt.show()

    # Plot pitch difference
    plt.figure(figsize=(10, 4))
    plt.plot(f0_difference, label="Pitch Difference (Hz)")
    plt.legend()
    plt.title("Pitch Differences Between Emotional and Neutral Speech")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Difference (Hz)")
    plt.show(block=True)

    pass


if __name__ == '__main__':
    spectrogram_difference()


    # test_audio_map()

    compare_words(1, 2)

    seg_audio_stft()
    segment_audio_into_words()

    print('sladf')
    # a = mp3_wav(path='test_n_1.mp3')
    rate_n1, data_n1 = read("test_n_1.wav")
    rate_e1, data_e1 = read("test_Angry_1.wav")
    rate_e2, data_e2 = read("test_Angry_2.wav")
    print(f"number of channels = {data_e1}")
    length1 = data_n1.shape[0] / rate_n1
    length2 = data_e1.shape[0] / rate_e1
    length3 = data_e2.shape[0] / rate_e2
    m_l = max([length1, length2, length3])
    print(f"length = {m_l}s")
    # plotting the audio time and amplitude
    time1 = np.linspace(0., length1, data_n1.shape[0])
    time2 = np.linspace(0., length2, data_e1.shape[0])
    time3 = np.linspace(0., length3, data_e2.shape[0])

    plt.plot(time1, data_n1, label="Audio")
    plt.plot(time2, data_e1, label="Audio")
    plt.plot(time3, data_e2, label="Audio")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show(block=True)

    # --- ---
    pitch_n1 = audio_to_pitch_over_time(path='test_n_1.wav')

    plt.plot(list(range(len(pitch_n1))), pitch_n1, label="Pitch neutral 1")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Pitch")
    plt.show(block=True)

    pass

    # Todo:
    #
    # 1. idea: 1. split audios on word level into chunks
    #          2. for each word chunk get peaks and rank them
    #          3. pair same words from two audios, map peaks with a same rank together
    #          4. calculate the differences in level and pitch for those peaks
    # 2. idea: 1. split audios on word level via transcription into chunks (fix separations if needed)
    #          2. normalize audios, set both to the same speed
    #          3. calculate curve over all peaks of audios (interpolate if needed -> same frequency) repr. max volume
    #          4. calculate pitches of audios
    #          5. calculate difference in pitches
    #          6. calculate difference between both audio peak-curves

    pass
