from BMSParserInterface import *
from MongoDBInterface import *
import json
import os
import hashlib
import traceback
from ffmpy import FFmpeg
import subprocess
from tqdm import tqdm

megabytes = 1048576

def anotherFileInSameDirectory(source,target):
    pos = source.rfind("/")
    pos = pos + 1 # for the / to get in the path
    return source[:pos] + str(target)

SOUND_EXTS = [".wav", ".ogg", ".mp3"]
IMAGE_EXTS = [".bmp", ".png", ".jpg", ".jpeg", ".gif"]

BMS_EXTS = [".bms",".bme",".bml"]

# Use the extension from alts to generate a group of file names with different extensions
def alternativeFile(path,alts):
    pos = path.rfind(".")
    result = []
    for alt in alts:
        result.append(path[:pos]+str(alt))
    return result

def firstFileExist(paths):
    for path in paths:
        if os.path.isfile(path):
            return path
    return None

def getExtension(path):
    try:
        pos = path.rfind(".")+1
        return path[pos:]
    except:
        #maybe the file has no extension
        return ""

def importBMStoDB(path):
    bi = BMSImporter(path)
    playables = bi.getAllPlayables()
    nonplayables = bi.nonplayables
    samples = bi.sampleFiles
    header = self.getHeader()
    raw = self.data
    document ={
        '1':2
    }

class BMSImporter:
    def __init__(self,path=None):
        self.data = None
        self.playables = None
        self.samples = None
        self.path = path
        self.sampleFiles = {}
        self.sampleFilesInverse = None
        self.nonplayables = None
        self.BMShash = None
        if path is not None:
            self.importBMS(path)
        else:
            print("Warning: Require .importBMS() to initialize.")

    def importBMS(self,path):
        with open(path, 'rb') as f:
            m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)  # File is open read-only
            data = m.read()
            BMShash = hashlib.sha1(data).hexdigest()
            self.BMShash = BMShash
        self.data = BMSParserInterface.getBMSInfo(path)
        #print(json.dumps(self.data,indent=4,sort_keys=True))

    def getAllEvents(self):
        return self.data['timednotes']

    def getEventKeysoundFilename(self,note):
        try:
            return self.getSampleWithID(note['data']['keysound'])
        except:
            print("Failed to get keysound filename for a note. Probably keysoundless BMS?")
            traceback.print_exc()
            return None


    #populates nonplayables too.
    def getAllPlayables(self):
        if self.playables == None:
            #print("Initializing playables...")
            self.playables = []
            self.nonplayables = []
            for note in self.getAllEvents():
                if 'column' in note['data']:
                    self.playables.append(note)
                else:
                    self.nonplayables.append(note)


            for note in self.playables:
                keysound_filename = self.getEventKeysoundFilename(note)
                time = note['time']
                true_sample_location = anotherFileInSameDirectory(self.path, keysound_filename)
                locations = alternativeFile(true_sample_location, SOUND_EXTS)
                true_sample_location = firstFileExist(locations)
                try:
                    hash = self.sampleFilesInverse[true_sample_location]
                except:
                    print("Can't find hash for a keysound. Maybe it's keysoundless.")
                    hash = None
                note['hash'] = hash

            for note in self.nonplayables:
                keysound_filename = self.getEventKeysoundFilename(note)
                time = note['time']
                true_sample_location = anotherFileInSameDirectory(self.path, keysound_filename)
                locations = alternativeFile(true_sample_location, SOUND_EXTS)
                true_sample_location = firstFileExist(locations)
                # print("Note at time "+str(time)+" have keysound filename "+true_sample_location)
                try:
                    hash = self.sampleFilesInverse[true_sample_location]
                except:
                    print("Can't find hash for a keysound. Maybe it's keysoundless.")
                    hash = None
                note['hash'] = hash
        #print("OK!")
        return self.playables

    def getRawBMS(self):
        return self.data['bmstext']

    def getHeader(self):
        return self.data['chart']['headers']

    def getHeadersWithoutWav(self):
        result = {}
        header = self.getHeader()
        for k in header["_data"]:
            if not k.startswith('wav') and not k.startswith('bmp'):
                result[k] = header["_data"][k]
        return result

    # for self.sampleFiles, only 1 file with such hash are recorded.
    # for self.sampleFilesInverse, all files are recorded.
    def getAllAudioSamples(self):
        if self.samples is None:
            sameHashSamples = {}
            header = self.getHeader()
            result = {}
            for k in header["_dataAll"]:
                if k.startswith('wav'):
                    keysoundID = k[3:].lower() #remove 'wav'
                    keysound_filename = header["_dataAll"][k][0]
                    result[keysoundID] = keysound_filename # for no reason we need [0]
                    true_sample_location = anotherFileInSameDirectory(self.path, keysound_filename)
                    locations = alternativeFile(true_sample_location, SOUND_EXTS)
                    true_sample_location = firstFileExist(locations)

                    if true_sample_location is None:
                        # Bad keysound, ignore this entry.
                        #print("Bad keysound at "+str(k)+" "+keysound_filename)
                        continue
                    statinfo = os.stat(true_sample_location)
                    if os.stat(true_sample_location).st_size < 1:
                        print("Empty keysound file detected:"+true_sample_location)
                        continue
                    with open(true_sample_location, 'rb') as f:
                        m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)  # File is open read-only
                        # Proceed with your code here -- note the file is already in memory
                        # so "readine" here will be as fast as could be
                        data = m.read()
                        hash = hashlib.sha1(data).hexdigest()
                        # print(hash)
                        if hash in self.sampleFiles:
                            sameHashSamples[true_sample_location] = hash
                        else:
                            self.sampleFiles[hash] = true_sample_location
                        #print("Note at time " + str(time) + " have keysound hash " + hash)
            self.sampleFilesInverse = {v: k for k, v in self.sampleFiles.items()}
            for k in sameHashSamples:
                self.sampleFilesInverse[k] = sameHashSamples[k]
            self.samples = result
        return self.samples

    def getSampleWithID(self,id_raw):
        id = id_raw.lower()
        if self.samples is None:
            self.getAllAudioSamples()
        if id in self.samples:
            return self.samples[id]
        else:
            print("Non-keysound BMS? Bad Sample ID hit: "+str(id))
            return None

    def dump(self):
        rawtext = self.getRawBMS()
        rawhash = self.BMShash
        headers = self.getHeadersWithoutWav()
        playables = self.getAllPlayables()
        nonplayables = self.nonplayables # need to do this after getting playables or it's []
        try:
            title = self.getHeader()['_dataAll']['title'][0]
        except:
            title = "Unknown title"
        try:
            artist = self.getHeader()['_dataAll']['artist'][0]
        except:
            artist = "Unknown artist"
        return {
            'title':title,
            'artist':artist,
            'hash':rawhash,
            'rawtext':rawtext,
            'headers':headers,
            'playables':playables,
            'nonplayables':nonplayables
        }

    def dumpSampleList(self):
        self.getAllAudioSamples()
        return self.sampleFiles

class BMSDBAdapter:
    def __init__(self):
        db = MongoDBInterface()
        db.setConnectionParam('143.215.206.155',27017) #fixed for now
        if db.open():
            self.db = db
        else:
            print("Failed to connect to DB!")
            self.db = None

    def dumpWholeDirectory(self,path):
        first_hash = None
        for file in os.listdir(path):
            for exts in BMS_EXTS:
                if file.endswith(exts):
                    this_file = os.path.join(path, file)
                    print("Dumping "+str(os.path.join(path, file)))
                    bi = BMSImporter(this_file)
                    this_dump = bi.dump()
                    if first_hash is None:
                        first_hash = this_dump['hash']
                    this_dump_samples = bi.dumpSampleList()
                    this_dump['parent'] = first_hash
                    self.db.insert(this_dump)
                    total_new = 0
                    total_updated = 0
                    total_large = 0
                    for k in tqdm(this_dump_samples):
                        statinfo = os.stat(this_dump_samples[k])
                        if statinfo.st_size > 2 * megabytes:
                            total_large = total_large + 1
                            ff=FFmpeg(
                                inputs={'pipe:0': None},
                                outputs = {'pipe:1': '-loglevel panic -f ogg -acodec libvorbis'}
                            )
                            stdout,stderr = ff.run(input_data=open(this_dump_samples[k],'rb').read(),stdout=subprocess.PIPE)
                            result = self.db.insertFile_memory(stdout,'ogg')
                            total_updated += result.matched_count
                            total_new += 1-result.matched_count
                        else:
                            result = self.db.insertFile(this_dump_samples[k],getExtension(this_dump_samples[k]))
                            total_updated += result.matched_count
                            total_new += 1-result.matched_count
                    print("Samples to database done, "+str(total_updated)+" old, "+str(total_new)+" new, "+str(total_large)+" large file encoded.")
