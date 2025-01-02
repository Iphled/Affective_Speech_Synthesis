import math
from os import listdir
from os.path import isfile, join

from sympy.physics.units import length

import Audio_to_Values

sentences={"IEO":"It's eleven o'clock.",
        "TIE":"That is exactly what happened.",
        "IOM":"I'm on my way to the meeting.",
        "IWW":"I wonder what this is about.",
        "TAI":"The airplane is almost full.",
        "MTI":"Maybe tomorrow it will be cold.",
        "IWL":"I would like a new alarm clock.",
        "ITH":"I think I have a doctor's appointment.",
        "DFA":"Don't forget a jacket.",
        "ITS":"I think I've seen this before.",
        "TSI":"The surface is slick.",
        "WSI":"We'll stop in a couple of minutes."}
onlyfiles = [f for f in listdir("data/AudioWAV") if isfile(join("data/AudioWAV", f))]
n=0

for filename in onlyfiles:
    sentence = sentences[filename.split("_")[1]]
    pitch=Audio_to_Values.audio_to_pitch_over_time("data/AudioWAV/"+filename)
    level,time=Audio_to_Values.audio_to_volume_over_time("data/AudioWAV/"+filename)

    filename2=filename.replace(".wav","")
    fp = open("data/curve_small/"+filename2, 'w')
    lstr=""
    pstr=""
    if len(level)<1000:
        for l in level:
            lstr=lstr+str(l)+","
    else:
        length=int(len(level)/1000)
        for i in range(1000):
            mittel=0
            count=0
            for j in range(i*length,(i+1)*length):
                if j<len(level):
                    count=count+1
                    m=level[j]
                    mittel=mittel+10**((level[j])/10)
            if (mittel==0):
                lstr=lstr+str(level[j])+","
            else:
                lstr=lstr+str(10*math.log(mittel/count,10))+","

    for l in pitch:
        pstr=pstr+str(l)+","
    fp.write(sentence+"\n")
    fp.write(lstr+"\n")
    fp.write(str(time)+"\n")
    fp.write(pstr)
    fp.close()
    n=n+1
    print(n)
