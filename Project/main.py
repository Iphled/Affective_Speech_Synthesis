import tkinter as tk
from fileinput import filename
from tkinter import ttk
from tkinter import filedialog
import os
from playsound import playsound

from Project.Audio_to_Values import audio_to_volume_over_time, audio_to_pitch_over_time
from Project.DL_querying import execute
# from skimage.morphology.misc import funcs

from Project.Emotion_extraction import extract_from_text, index_from_emotion
from Project.Text_to_speech import text_to_speech
from Project.transform_all_audios import pitch
from Project.write_back_audio import write_back_audio

emotion = "neutral"
emotion_values = ("Neutral", "Joy", "Sadness", "Anger", "Fear")
audio=None

def synthesize():
    global audio
    text = textvar.get()
    emotion = combobox.get()
    audio = text_to_speech(text)
    if emotion is not "Neutral" and audio is not None:
        filename = 'audio_tmp' + '.mp3'
        audio.save(filename)
        level,time = audio_to_volume_over_time(filename,True)
        pitch=audio_to_pitch_over_time(filename)
        time,volume,pitch=execute(level,pitch,emotion)
        audio=write_back_audio(time,volume,pitch,audio,time)
    play_audio()
    pass


def play_audio():
    global audio
    print(audio)
    if audio is not None:
        filename = 'audio_tmp' + '.mp3'
        audio.save(filename)
        playsound(filename)
        os.remove(filename)

def savetofile():
    global audio
    print(audio)
    if audio is not None:
        f = filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("mp3 files","*.mp3"),("all files","*.*")))
        audio.save(f)


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
    combobox = ttk.Combobox(root, state="readonly", values=emotion_values)
    combobox.pack()
    label = tk.Label(root, text="")
    label.pack()

    button = tk.Button(root, text="Synthesize", command=synthesize)
    button.pack()

    savebutton = tk.Button(root, text="SavetoFile", command=savetofile)
    savebutton.pack()
    root.mainloop()
