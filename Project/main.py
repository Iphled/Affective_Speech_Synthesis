import tkinter as tk
import wave
from tkinter import ttk
from tkinter import filedialog
import os

import librosa
from playsound import playsound
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io.wavfile import read

from Project.Audio_to_Values import audio_to_volume_over_time, audio_to_pitch_over_time
# from skimage.morphology.misc import funcs

from Project.Emotion_extraction import extract_from_text, index_from_emotion
from Project.Text_to_speech import text_to_speech
# from Project.write_back_audio import write_back_audio, writeback_happy, writeback_sad, writeback_angry, \
#     writeback_fearful
from Project.emotional_xtts_pipeline import convert_text_to_audio_and_store
from Project.denoiser import denoise_audio, restore_volume

emotion = "neutral"
emotion_values = ("Neutral", "Joy", "Sadness", "Anger", "Fear")
syn_methods = ("Acoustic Features", "Emotional XTTS-v2")
audio=None


def synthesize():
    global audio
    text = textvar.get()
    emotion = combobox.get()
    audio = text_to_speech(text)
    filename,audio,length=gtts_to_audiosegment(audio)
    if emotion!="Neutral" and audio is not None:
        if emotion=="Joy":
            audio=writeback_happy(audio,length)
        if emotion=="Sadness":
            audio=writeback_sad(audio,length)
        if emotion=="Anger":
            audio=writeback_angry(audio,length)
        if emotion=="Fear":
            audio=writeback_fearful(audio,length)
        os.remove(filename)

    play_audio()
    pass


def synthesize_xtts():
    global audio
    text = textvar.get()
    emotion = combobox.get()
    emotion_map = {"Neutral": 'NEU', "Joy": 'HAP', "Sadness": 'SAD', "Anger": 'ANG', "Fear": 'FEA'}
    # convert text into emotional audio
    convert_text_to_audio_and_store(text=text,
                                    output_file_path='tmp_app_xtts_audio.wav',
                                    gender='female',
                                    emotion=emotion_map[emotion])
    # reduce noise from audio
    emo_audio, sr = librosa.load('tmp_app_xtts_audio.wav', sr=16000)
    denoised_audio = denoise_audio(emo_audio, sample_rate=sr)
    # restore volume to match original audio
    denoised_audio_restored = restore_volume(denoised_audio, reference=emo_audio, window_size=4096)
    os.remove('tmp_app_xtts_audio.wav')
    audio = denoised_audio_restored

    play_audio()
    pass


def call_syn_method():
    method = syn_method_selector.get()
    if method == 'Acoustic Features':
        synthesize()
    elif method == 'Emotional XTTS-v2':
        synthesize_xtts()
    else:
        return


def gtts_to_audiosegment(audio):
    filename = 'audio_tmp' + '.wav'
    filename2 = 'audio_tmp' + '.mp3'
    audio.save(filename2)
    sound = AudioSegment.from_mp3(filename2)
    sound.export(filename, format="wav")
    rate, data = read(filename)
    length = data.shape[0] / rate
    os.remove(filename2)
    return filename,sound,length


def play_audio():
    global audio
    print(audio)
    if audio is not None:
        filename = 'audio_tmp' + '.mp3'
        if type(audio) == AudioSegment:
            audio.export(filename, format="mp3")
        else:
            audio.save(filename)
        playsound(filename)
        os.remove(filename)

def savetofile():
    global audio
    print(audio)
    if audio is not None:
        f = filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("mp3 files","*.mp3"),("all files","*.*")))
        audio.export(f,format="mp3")


def find_emotion():
    global emotion
    if "readonly" in combobox.state():
        combobox.configure(state='disabled')
        emotion = extract_from_text(textvar.get())
        print(emotion)
        combobox.current(index_from_emotion(emotion, emotion_values))
    else:
        combobox.configure(state='readonly')


def change_text():
    checkbox.deselect()
    combobox.configure(state='readonly')


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("500x200")
    root.title("Affective Speech Synthesizer")
    label = tk.Label(root, text="Sentence:")
    label.pack()
    textvar = tk.StringVar()
    textvar.trace("w", lambda name, index, mode, tv=textvar: change_text())
    textfield = tk.Entry(root, width=50, textvariable=textvar)
    textfield.pack()
    checkbox = tk.Checkbutton(root, text="Extract Emotion Automatically", command=find_emotion)
    checkbox.pack()

    emo_frame = tk.Frame(root)
    emo_frame.pack()
    combobox = ttk.Combobox(emo_frame, state="readonly", values=emotion_values)
    combobox.pack(side='right', anchor='e', expand=True)
    combobox.set('Neutral')
    emotion_label = tk.Label(emo_frame, text='Emotion:')
    emotion_label.pack(side='left', anchor='w', expand=True)

    method_frame = tk.Frame(root)
    method_frame.pack()
    syn_method_selector = ttk.Combobox(method_frame, state="readonly", values=syn_methods)
    syn_method_selector.pack(side='right', anchor='e', expand=True)
    syn_method_selector.set('Acoustic Features')
    method_label = tk.Label(method_frame, text='Method:')
    method_label.pack(side='left', anchor='w', expand=True)
    label = tk.Label(root, text="")
    label.pack()

    button = tk.Button(root, text="Synthesize", command=call_syn_method)
    button.pack()

    savebutton = tk.Button(root, text="Save to File", command=savetofile)
    savebutton.pack()
    root.mainloop()
