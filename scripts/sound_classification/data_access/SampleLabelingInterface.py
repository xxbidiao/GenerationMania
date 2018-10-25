import pymongo
import sys
from pymongo import MongoClient
import hashlib
import mmap
from tqdm import tqdm
from MongoDBInterface import *
import base64

class SampleLabelingInterface:
    def __init__(self):
        self.mi = MongoDBInterface()
        self.mi.open()
        self.classifier = None


    def setClassifier(self,obj):
        self.classifier = obj

    def classifyAll(self):
        for result in tqdm(mi.getFileWithoutLabel()):
            data = result['data']
            ext = result['ext']
            hash = result['hash']
            label = self.classifier.classify(data,ext) # "wav" "ogg" "mp3"
            self.mi.updateLabelForSample(hash,label)

    # Just in case labels are wrongly labeled...
    # very quick operation.
    def removeAllLabels(self):
        self.mi.removeAllLabel()


class Classifier:
    def __init__(self):
        pass

    def classify(self,data,ext):
        pass

class NullClassifier(Classifier):

    def classify(self,data,ext):
        return {'labelnumber':1234,'label':'bass','confidence':0.8}




# Testing connections, showing first 10 entries in the sample database.

mi  = MongoDBInterface()
mi.open()
count = 0
#mi.removeAllLabel()
classifier = NullClassifier()
print(mi.host)
for result in tqdm(mi.getFileWithoutLabel()):
    if count > 0:
        print("Stop at 10 iterations for now.")
        break
    print(result['hash'])
    #(base64.b64encode(result['data']))
    test_file = file('sample.ogg', 'w')
    test_file.write(result['data'])
    test_file.close()
    print(result['ext'])
    print("Class:"+str(classifier.classify(result['data'],result['ext'])))
    print("Write to DB disabled for now. Uncomment the next line.")
    #mi.updateLabelForSample(result['hash'],'test_label')
    count = count + 1
print(count)
