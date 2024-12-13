import tkinter as tk
from tkinter import ttk

from skimage.morphology.misc import funcs

from Project.Emotion_extraction import extract_from_text, index_from_emotion
from Project.Text_to_speech import text_to_speech

emotion="neutral"
emotion_values=("Neutral","Joy","Sadness","Anger","Fear")

def synthesize():
    text=textvar.get()
    emotion=combobox.get()
    audio=text_to_speech(audio)


def find_emotion():
    if "readonly" in combobox.state():
        combobox.configure(state='disabled')
    else:
        combobox.configure(state='readonly')
    emotion=extract_from_text(textvar.get())
    combobox.current(index_from_emotion(emotion,emotion_values))

if __name__=="__main__":
    root = tk.Tk()
    root.geometry("500x300")
    root.title("Affective Speech Synthesizer")
    label = tk.Label(root,text="Sentence")
    label.pack()
    textvar = tk.StringVar()
    textfield = tk.Entry(root,width=50,textvariable=textvar)
    textfield.pack()
    checkbox = tk.Checkbutton(root,text="Extract Emotion Automatically",command=find_emotion)
    checkbox.pack()
    combobox= ttk.Combobox(root,state="readonly",values=emotion_values)
    combobox.pack()

    button =tk.Button(root,text="Synthesize",command=synthesize)
    button.pack()
    root.mainloop()