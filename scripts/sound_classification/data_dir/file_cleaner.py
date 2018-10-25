import os
from pydub import AudioSegment
cwd = os.getcwd()
categories = ['piano',
'bass',
'synth',
'al',
'guitar',
'kick',
'snare',
'on',
'drum',
'full',
'violin',
'fx',
'acid',
'harp',
'rap',
'hard',
'glitch',
'vocal',
'beat',
'hop',
'breakbeat',
'lute',
'breakbeats',
'noise',
'nu',
'and',
'idm',
'cutup',
'accordion',
'marimba']

for category in categories:
    if not os.path.exists(cwd + "/data_dir/" + category):
        os.makedirs(cwd + "/data_dir/" + category)
testing_list = open("testing_list.txt", "w")
for path, subdirs, files in os.walk("./"):
    for fn in files:
        cwd = path
        for category in categories:
            if category in fn:
                try:
                    print(cwd + "/" + category + "/" + fn)
                    sound = AudioSegment.from_file(cwd + "/" + fn)
                    sound.set_channels(1)
                    sound.channels = 1
                    sound.set_frame_rate(16000)
                    sound.frame_rate = 16000
                    sound.export("./data_dir/" + category + "/" + fn[:-4] + ".wav", format="wav")
                    break
                except Exception as e:
                    print "Error with " + cwd + "/" + category + "/" + fn

            testing_list.write(category + "/" + fn[:-4] + ".wav" + "\n")

testing_list.close()
