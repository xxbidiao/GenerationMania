from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pymongo
import sys
from pymongo import MongoClient
import hashlib
import mmap
import base64
from tqdm import tqdm
import soundfile as sf
import sys
from pydub import AudioSegment

sys.path.insert(0, "../data_access/")

from ..MongoDBInterface import *
from .pitch_detection import get_pitch

class PitchLabelingInterface:
    def __init__(self):
        self.mi = MongoDBInterface()

    def removeAllLabels(self):
        self.mi = MongoDBInterface()
        self.mi.open()
        self.mi.removeAllPitches()

    def label(self, result, fname, ext):
        try:
            test_file = open(fname + ext, 'w')
            test_file.write(result['data'])
            test_file.close()
            sound = AudioSegment.from_file(fname + ext)
            sound.export(fname + "wav", format="wav")
            current_pitch = get_pitch(fname + "wav")
            self.mi.updatePitchForSample(result['hash'], current_pitch)
            print(current_pitch, result['hash'])
        except:
            print("Invalid audio sample: ", result['hash'])

    def label_all(self):
        """Entry point for script"""
        self.mi = MongoDBInterface()
        self.mi.open()
        for result in tqdm(self.mi.getFileWithoutPitch()):
            self.label(result, 'sample.', result['ext'])

    def label_one(self, data, ext):
        try:
            test_file = open('temp.' + result['ext'], 'wb')
            test_file.write(result['data'])
            test_file.close()
            return get_pitch('temp.' + ext)
        except:
            print("Error in labeling a single pitch")
            return ""

    def label_all_func(self):
        self.mi = MongoDBInterface()
        self.mi.open()
        print("Processing pitch samples...")
        total_processed = 0
        for result in tqdm(self.mi.getFileWithoutPitch()):
            try:
                self.label(result, 'sample.', result['ext'])
            except:
                print("Invalid audio sample: ", result['hash'])
        return total_processed

labelingInstance = PitchLabelingInterface()
def getLabelingInstance():
    return labelingInstance
