import librosa
import matplotlib.pyplot as plt
import os
import numpy as np
import noisereduce as nr
import soundfile as sf


def load_audio(file_path, sr=16000):
    y, _ = librosa.load(file_path, sr=sr)
    # y, _ = librosa.effects.trim(y, top_db=20)
    return y


def restore_volume(audio, reference, window_size=4096):
    """Restores volume of an audio segment-vise to the level of the reference audio

    :param audio: original audio wav signal, needs to be adjusted
    :param reference: audio wav signal, reference for volume
    :param window_size: segment length
    :return: audio with the wav form characteristic of the original audio with volume close to the reference volume
    """
    ref_len = reference.shape[0]
    audio_parts = []
    for i in range(0, ref_len, window_size):
        ref_peak = np.max(reference[i:i+window_size])
        audio_peak = np.max(audio[i:i+window_size])
        adjust_factor = ref_peak / audio_peak
        audio_parts.append(audio[i:i+window_size] * adjust_factor)
    new_audio = np.concatenate(audio_parts)
    return new_audio


def denoise_audio(audio, sample_rate):
    """Reduces background noise from audio.

    :param audio: audio wav signal
    :param sample_rate: sample rate of audio
    :return: audio wav signal with reduced noise
    """
    return nr.reduce_noise(y=audio, sr=sample_rate)


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR = ROOT_DIR + '\\Benchmark_wav\\easy'

    audio = load_audio(file_path=AUDIO_DIR + '\\ANG\\FEMALE_05_ANG.wav')

    # denoise audio
    denoised_audio = denoise_audio(audio, sample_rate=16000)
    sf.write('denoised_angry.wav', denoised_audio, 16000)

    denoised_vol = restore_volume(denoised_audio, audio, window_size=4000)
    sf.write('denoised_vol_angry.wav', denoised_vol, 16000)

    plt.figure(figsize=(12, 6))
    plt.subplot(311)
    plt.plot(audio, label='Original')
    plt.legend()
    plt.subplot(312)
    plt.plot(denoised_audio, label='Denoised')
    plt.legend()
    plt.subplot(313)
    plt.plot(denoised_vol, label='Denoised With Restored Volume')
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)
    pass
