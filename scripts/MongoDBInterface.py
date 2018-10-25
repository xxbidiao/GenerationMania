#MongoDBInterface.py
# Interface for using MongoDB.

import pymongo
import sys
from pymongo import MongoClient
import hashlib
import mmap
from .Config import config

main_db = config.main_db
binary_collection = config.binary_collection
chart_collection = config.chart_collection

default_host = config.default_host
default_port = config.default_port

class MongoDBInterface:
    def __init__(self):
        self.host = None
        self.port = None
        self.client = None
        self.database = None #database object
        self.binaryDB = None

    # Set parameters of future connections.
    def setConnectionParam(self,host,port):
        self.host = host
        self.port = port

    # Open connections.
    def open(self):
        if self.host is None:
            self.host = default_host
        if self.port is None:
            self.port = default_port
        try:
            self.client = MongoClient(self.host,self.port)
            self.database = self.client[main_db][chart_collection]
            self.binaryDB = self.client[main_db][binary_collection]
            clientinfo = self.client.server_info()
            return True
        except:
            print("Failed to open connection.")
            print(sys.exc_info()[0])
            return False

    # Insert an object, assuing it has a field of `hash`.
    def insert(self,obj):
        a = self.database.update_one({'hash':obj['hash']},{'$set':obj},upsert=True)

    # Get charts based on criteria. If criteria is None, get all Charts.
    def getChartDocument(self,criteria):
        return self.database.find(criteria,no_cursor_timeout=True)

    # Get audio samples based on criteria. If criteria is None, get all Charts.
    def getSampleDocument(self,criteria):
        return self.binaryDB.find(criteria,no_cursor_timeout=True)

    # Load a file into memory, then call `insertFile_memory`.
    def insertFile(self,path,ext=None):
        with open(path,'rb') as f:
            m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)  # File is open read-only
            # Proceed with your code here -- note the file is already in memory
            # so "readine" here will be as fast as could be
            data = m.read()
            return self.insertFile_memory(data,ext)

    #Insert an audio file from memory to the database.
    def insertFile_memory(self,data,ext):
        hash = hashlib.sha1(data).hexdigest()
        return self.binaryDB.update_one({'hash': hash}, {'$set': {'ext':ext,'hash': hash, 'data': data}}, upsert=True)


    def getFileWithoutLabel(self):
        # TODO: check the root cause of this causing timeouts
        return self.binaryDB.find({"label":{"$exists":False}},no_cursor_timeout=True)

    def getFileWithoutPitch(self):
        return self.binaryDB.find({"pitch":{"$exists":False}},no_cursor_timeout=True)

    def updateLabelForSample(self,hash,label):
        return self.binaryDB.update({'hash':hash},{'$set':{'label':label}})

    def getLabelForSample(self,hash):
        return self.binaryDB.find_one({'hash':hash})['label']

    def getAllFiles(self):
        return self.binaryDB.find(no_cursor_timeout=True)

    def removeAllLabel(self):
        return self.binaryDB.update_many(
            {},
            {'$unset':{'label':''}},
        )

    def getFile(self,hash):
        try:
            result = self.binaryDB.find_one({"hash":hash})
            return result['data']
        except:
            return None


def test_database():
    db = MongoDBInterface()
    db.setConnectionParam(default_host,default_port)
    if db.open():
        print("Database Test PASS.")
    else:
        print("Database Test FAIL.")

def getAllFiles():
    db = MongoDBInterface()
    db.setConnectionParam(default_host,default_port)
    if db.open():
        return db.getAllFiles()
