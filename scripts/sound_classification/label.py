from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from pydub import AudioSegment

cwd = os.getcwd()
for fn in os.listdir('.'):
    if os.path.isfile(fn) and "sample" in fn:
        sound = AudioSegment.from_file(fn)
        sound.set_channels(1)
        sound.channels = 1
        sound.set_frame_rate(16000)
        sound.frame_rate = 16000
        sound.export(fn[:-4] + ".wav", format="wav")
        execfile('label_wav.py')
