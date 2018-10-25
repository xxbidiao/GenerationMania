import pymongo
import sys
from pymongo import MongoClient
import hashlib
import mmap

main_db = 'musicgame'
binary_collection = 'audiosamples'
chart_collection = 'charts'

default_host = '143.215.206.155'
default_port = 1337

class MongoDBInterface:
    def __init__(self):
        self.host = None
        self.port = None
        self.client = None
        self.database = None #database object
        self.binaryDB = None

    def setConnectionParam(self,host,port):
        self.host = host
        self.port = port

    def open(self):
        if self.host is None:
            self.host = default_host
        if self.port is None:
            self.port = default_port
        try:
            self.client = MongoClient(self.host,self.port)
            self.database = self.client[main_db][chart_collection]
            self.binaryDB = self.client[main_db][binary_collection]
            return True
        except:
            print("Failed to open connection.")
            print(sys.exc_info()[0])
            return False

    def insert(self,obj):
        a = self.database.update_one({'hash':obj['hash']},{'$set':obj},upsert=True)

    def insertFile(self,path,ext=None):
        with open(path,'rb') as f:
            m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)  # File is open read-only
            # Proceed with your code here -- note the file is already in memory
            # so "readine" here will be as fast as could be
            data = m.read()
            return self.insertFile_memory(data,ext)

    def insertFile_memory(self,data,ext):
        hash = hashlib.sha1(data).hexdigest()
        return self.binaryDB.update_one({'hash': hash}, {'$set': {'ext':ext,'hash': hash, 'data': data}}, upsert=True)

    def getFileWithoutLabel(self):
        return self.binaryDB.find({"label":{"$exists":False}})

    def updateLabelForSample(self,hash,label):
        return self.binaryDB.update({'hash':hash},{'$set':{'label':label}})

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



#
# # test code
# db = MongoDBInterface()
# db.setConnectionParam('localhost',27017)
# if db.open():
#     #db.insertFile('MongoDBInterface.py')
#     result = db.getFile("45791226e0664ea52b5128ed1b85ddae8ab9913c")
#     print(result)
#     #a = {'first':2,'second':{'a':4,'b':6}}
#     #db.insert(a)
# else:
#     print("Error.")
#
