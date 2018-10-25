# createChart.py
# Creates .bmson chart as exported format.

from .BMStoDB import BMSImporter
from .model_feedforward import eval_model as playable_eval
from .model_feedforward import anti_tr as prepare_f
from .model_ff_cols import eval_model as column_eval
from .model_ff_cols import anti_tr as prepare_c
from .DBtoLearn import default_class,possible_audio_classes,total_classes,beat_to_frame,addNote,column_to_number,aggregateBeatsToMeasures,look_back
from .DifficultyCalculation import DifficultyCalculator
from .BMSONAdapter import generate_bmson,bmson
from .data_utils import get_all_files
from ffmpy import FFmpeg
from multiprocessing import Pool, TimeoutError
import torch
import os, subprocess
import numpy as np
from .AudioInterface import do_one_audio,do_batch_ogg
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from shutil import copyfile

# If True, all notes will be classified as a dummy class.
# Good for testing other part of the program.
debug = False

dim_note_data = 304

#some in-line functions

# Stub for using FFmpeg to convert file into format that audio classifier requires.
def convert_ffmpeg(infile):
    ff = FFmpeg(
        inputs={'pipe:0': None},
        outputs={'pipe:1': '-loglevel panic -f ogg -acodec libvorbis'}
    )
    file_content = open(infile, 'rb').read()
    stdout, stderr = ff.run(input_data=file_content, stdout=subprocess.PIPE)
    return stdout

# Tests for whether `obj1` and `obj2` are in the same column (and are playable).
def is_in_same_column(obj1,obj2):
    if obj1['gen_column'] < 0:
        return False
    if obj1['gen_column'] == obj2['gen_column']:
        return True
    return False

# Returns how much time in seconds obj2 is later than obj1.
def time_difference(obj1,obj2):
    return obj2['time'] - obj1['time']


class ChartCreator:
    def __init__(self,path):
        self.file_path = path
        self.write_path = None
        self.main_style = None
        if path is not None:
            self.extract_style()

    # Extract style information from currently imported chart.
    def extract_style(self):
        self.main_style = ChartStyle(self.file_path)
        self.main_style.parse_chart()
        self.main_style.create_representation()
        return self.main_style

    # use the provided `style` and current audio information to create a chart.
    def generate_chart(self,style):
        playable_value = playable_eval(prepare_f(style.style))
        column_value = column_eval(prepare_c(style.style))
        assignment_value = []
        for item in range(len(playable_value)):
            if torch.argmax(playable_value[item]) == 0:
                #playable
                assignment = int(torch.argmax(column_value[item]))
            else:
                assignment = -1
            assignment_value.append(assignment)
        return assignment_value

    #use main style to create a chart.
    #Assuming main style is created.
    def generate_chart_using_main_style(self):
        assignment_result = self.generate_chart(self.main_style)
        objects = []
        for index,value in enumerate(assignment_result):
            note = self.main_style.get_note_at_index(index)
            objects.append(note.copy())
            objects[index]['gen_column'] = assignment_result[index]
        self.post_process(objects)
        return generate_bmson(objects,self.main_style.header)

    # Move notes around so that they don't come in extremely small intervals.
    def post_process(self,objects,options = None):
        move_overlapping_notes = True
        objects.sort(key=lambda item:item['time'])
        if move_overlapping_notes:
            for index in range(len(tqdm(objects))):
                considering_object = []
                index2 = index + 1
                this_obj = objects[index]
                try:
                    other_obj = objects[index2]
                except IndexError:  # no more thing to consider, the whole process should stop
                    break
                while time_difference(this_obj,other_obj) < 0.3:
                    if other_obj['gen_column'] >= 0:
                        considering_object.append(other_obj)
                    index2 += 1
                    try:
                        other_obj = objects[index2]
                    except IndexError: # no more thing to consider
                        break
                occupied = {}
                occupied[this_obj['gen_column']] = True
                for item in considering_object:
                    orig_col = item['gen_column']
                    if orig_col in occupied:
                        candidate = [8-orig_col,orig_col+1,orig_col+2,orig_col+3,orig_col+4]
                        for index2,value2 in enumerate(candidate):
                            candidate[index2] = value2 % 8
                        for selection in candidate:
                            if selection not in occupied:
                                occupied[selection] = True
                                item['gen_column'] = selection
                                print("Note %d moved from %d to %d."%(item['index'],orig_col,selection))
                                break
    # Build a chart using provided parts.
    def assemble_chart(self,headers=None,objects=None,audio=None):
        result = {}
        result['headers'] = headers
        result['objects'] = objects
        result['audio'] = audio
        return result

# Data structure for representation of Chart Styles.
class ChartStyle:
    def __init__(self,path):
        self.file_path = path
        self.audio_data = {}
        self.object_data = {}
        self.doc = None
        self.style = None
        self.style_index = None
        self.bpm = -1
        self.header = None

    # Get note at position `index`.
    def get_note_at_index(self,index):
        try:
            def findidx(l,idx):
                return list(filter(lambda x: x['index'] == idx,l))
            result = findidx(self.doc['playables'],index)+findidx(self.doc['nonplayables'],index)
            assert len(result) == 1
            return result[0]
        except:
            print("invalid index:%d in get_note_at_index."%index)


    def generateCommonInfo(self,obj):
        info = {}
        hash = obj['hash']
        class_name = self.audio_data[hash]['class']
        try:
            class_number = possible_audio_classes[class_name]
        except:
            class_number = default_class
        time = obj['time']
        beat = obj['data']['beat']
        beat -= int(beat)
        beat = beat_to_frame(beat)  # cutoff is at 16th
        result = {'class': class_number, 'beat': beat, 'time': time}
        result['index'] = obj['index']
        return result

    # create a machine-readable representation (offline version of part of DBtoLearn
    # default to using self.doc (from parse_chart)
    def create_representation(self,document = None):
        if document is None:
            doc = self.doc
        else:
            doc = document
        diffc = DifficultyCalculator("osu")
        playable_counter = 0
        diff = diffc.calculate(doc)
        bpm = float(doc['headers']['bpm'])
        self.bpm = bpm
        start_token = [0] * dim_note_data
        start_token[0] = diff
        # all_result_for_chart.append([0] * dim_note_data) # "start" token
        notes = {}
        playables = doc['playables']
        nonplayables = doc['nonplayables']
        max_beat = 0
        note_list = {}
        bg_list = {}
        for obj in playables:
            beat = beat_to_frame(obj['data']['beat'])
            if max_beat < beat:
                max_beat = beat
            addNote(note_list, beat, obj)
        for obj in nonplayables:
            beat = beat_to_frame(obj['data']['beat'])
            if max_beat < beat:
                max_beat = beat
            addNote(bg_list, beat, obj)
        aggregated_data = {}
        for beat_count in range(0, max_beat):
            aggregated_data[beat_count] = []
            for obj in note_list.get(beat_count, []):
                info = self.generateCommonInfo(obj)
                info['total_beat'] = beat_count
                info['playable'] = True
                info['column'] = column_to_number(obj['data']['column'])
                info['index'] = obj['index']
                aggregated_data[beat_count].append(info)
            for obj in bg_list.get(beat_count, []):
                info = self.generateCommonInfo(obj)
                info['total_beat'] = beat_count
                info['playable'] = False
                info['column'] = 90
                info['index'] = obj['index']
                aggregated_data[beat_count].append(info)
        flattened_data = []

        for beat_count in range(0, max_beat):
            this_beat = {}
            for item in aggregated_data[beat_count]:
                item_class = item['class']
                playable = item['playable']
                if item_class not in this_beat:
                    this_beat[item_class] = [0, 0]
                if playable:
                    this_beat[item_class][0] += 1

                else:
                    this_beat[item_class][1] += 1
            flattened_data.append(this_beat)

        flattened_aggregated_data = []
        for beat_count in range(0, max_beat):
            if len(aggregated_data[beat_count]) < 1:
                continue
            this_note_data = {0: aggregated_data[beat_count]}
            for factors in look_back:
                this_note_data[factors] = aggregateBeatsToMeasures(flattened_data, beat_count - factors, beat_count)
            flattened_aggregated_data.append(this_note_data)
        this_song = []
        this_song_object_index = []
        for raw_data in flattened_aggregated_data:
            for element in raw_data[0]:  # separate entry for every single object
                this_note_original_index = element['index']
                this_note_data = np.zeros(dim_note_data)
                this_note_data[0] = diffc.getOsuNoteStrain(element['time'])
                this_note_data[1] = element['beat']
                if element['playable'] is True:
                    this_note_data[2] = 1
                else:
                    this_note_data[3] = 1
                if element['playable'] is True:
                    playable_counter += 1
                this_note_data[4 + element['class']] = 1
                # if getOrNone(criteria, 'extra_info'):
                this_note_data[31] = int(element['total_beat']) + 0.01 * element['column']
                index = 32
                for factors in look_back:
                    for item in raw_data[factors]:
                        true_idx = index + item * 2
                        this_note_data[true_idx] = raw_data[factors][item][0]
                        this_note_data[true_idx + 1] = raw_data[factors][item][1]
                    index += total_classes * 2
                this_song.append(this_note_data)
                this_song_object_index.append(this_note_original_index)
        counter_this_song = 0
        for elem in this_song:
            if elem[2] == 1:
                counter_this_song += 1
        self.style = this_song
        self.style_index = this_song_object_index
        self.style = np.array(self.style)
        return self.style

    # parse a chart and get its audio and object data.
    def parse_chart(self):
        this_file = self.file_path
        #print("Dumping " + this_file)
        bi = BMSImporter(this_file)
        print("PASS 1 - Creating note profile...")
        this_dump = bi.dump()

        self.doc = this_dump
        self.header = this_dump["headers"]

        print("Done.")

        #hash:filepath
        this_dump_samples = bi.dumpSampleList()

        #unpacking dicts into arrays
        sample_keys = []
        sample_values = []
        for key in this_dump_samples:
            sample_keys.append(key) #hash
            sample_values.append(this_dump_samples[key]) #filepath

        print("PASS 2 - Loading audio samples...")
        classification_pending = {}
        # paralleling code
        pool = Pool(processes=16)
        print("Loading audio files...")
        sample_result = pool.map(convert_ffmpeg, sample_values, chunksize=8) #file in order of hash
        print("Done.")
        print("Classifying audio files...")
        label_result = do_batch_ogg(sample_result)
        print("Done.")
        for i in range(len(sample_keys)):
            k = sample_keys[i]
            self.audio_data[k] = {'class': label_result[i], 'file': this_dump_samples[k]}
        print("All Done!")

    # Mix this style with another style `other`.
    # Overwrite the difficulty curve ([0]) and interpolates it.
    def mix_with(self,other):
        my_length = len(self.style)
        if my_length < 1:
            print("Style length = 0 encountered in mix_with")
            return False
        other_length = len(other.style)
        #take which cell?
        def take(my_index):
            result = int(np.floor(1.0 * my_index * other_length / my_length))
            if result == other_length: # the last note
                result = other_length - 1
            return result

        for row in range(len(self.style)):
            for col in range(len(self.style[row])):
                if col == 0: #copy difficulty
                    self.style[row][col] = other.style[take(row)][col]
                if col >= 32: #interpolation of style
                    self.style[row][col] = other.style[take(row)][col]
        return True

# Mix two charts using audio samples from `audio_provider` and style information from `style_provider`.
def mix_two_charts(audio_provider,style_provider):
    #path = "scripts/test/test/_71.bme"
    audio_chart = ChartCreator(audio_provider)
    #path2 = "scripts/test/test2/_7-01.bme"
    style_chart = ChartCreator(style_provider)
    audio_chart.main_style.mix_with(style_chart.main_style)
    return audio_chart.generate_chart_using_main_style()

# Creates whole bmson package, ready for play, by incluiding all audio files.
def create_bmson_package(bmson,audio_source):
    import json
    target_bmson = "genMania.bmson"
    bmson_result = json.dumps(bmson.doc)
    title = bmson.doc['info']['title']
    subfolder_name = "gen_"+title
    import os.path
    try:
        os.mkdir(subfolder_name)
    except Exception:
        print("[BMSON] Warning: A subfolder already exists for the following title, overwriting: %s"%title)
        pass
    # shortens the append subdirectory function
    def s(filename):
        return os.path.join(subfolder_name,filename)

    file = open(s(target_bmson), mode='w')
    file.write(bmson_result)
    file.close()
    print("BMSON File wrote to %s"%target_bmson)
    for element in bmson.doc["sound_channels"]:
        soundfile = element['name']
        filename, file_extension = os.path.splitext(soundfile)
        all_originals = get_all_files(os.path.dirname(audio_source),filename+".*")
        for options in all_originals:
            original_soundfile_location = os.path.join(os.path.dirname(audio_source),options)
            target_soundfile_location = s(options)
            print("Copying %s to %s"%(original_soundfile_location,target_soundfile_location))
            copyfile(original_soundfile_location,target_soundfile_location)
    print("Done! Use the folder %s in the same directory of main script to play."%(subfolder_name))