# DBtoLearn.py
# This file generates flattened data from object-based database for training purposes.
import numpy as np
from lru import LRU
import torch
from .BMSParserInterface import *
from .MongoDBInterface import *
import json
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from .DifficultyCalculation import *
import matplotlib.pyplot as plt
from .data_utils import *

# Padding seq length to this for batch training.
PAD_LENGTH = 2560
PAD_LENGTH_PLAYABLE_CLASSIFICATION = 10240

# Dimension of the features.
dim_note_data = 304

# Possible note segments in a measure. 4 means quarter notes, 8 means eighth, etc.
possible_alignment = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192, 256, 384, 768]

possible_audio_classes = {'piano': 0, 'synth': 1, 'bass': 2, 'guitar': 3, 'kick': 4, 'snare': 5, 'drum': 6
    , 'saw': 7, 'violin': 8, 'acid': 9, 'harp': 10, 'full': 11, 'fx': 12, 'rap': 13
    , 'cutup': 14, 'beat': 15, 'lute': 16, 'vocal': 17,
                          'glitch': 18, 'hard': 19, 'hop': 20, 'marimba': 21, 'clap': 22,
                          'accordion': 23, 'noise': 24, 'idm': 25, 'cymbal': 26}

# If something terrible wrong happens (very seldomly) this will be used as the label.
default_class = 24  # noise

# default normalization method.
normalize_method = 1

# Pad array to given length, keeping the 2nd dim intact, using all zeros
def pad_array(arr, pad_to=PAD_LENGTH):
    shape_of_arr = arr.shape
    dim0 = shape_of_arr[0]  # assuming
    new_shape = (pad_to, shape_of_arr[1])
    result = np.zeros(new_shape)
    result[:shape_of_arr[0], :shape_of_arr[1]] = arr
    return result

# Get element from dictionary using key `item`, if nothing found, return None.
def getOrNone(dict, item):
    if item in dict:
        return dict[item]
    return None

# Database instance.
dbi = None

# Sample cache to handle same samples being used over and over in the same Chart.
sample_cache = LRU(5000)

# Get audio class of a sample by its hash.
def getSampleClass(hash):
    global dbi
    if dbi is None:
        dbi = MongoDBInterface()
        dbi.open()
    if hash not in sample_cache:
        try:
            object_class = dbi.getLabelForSample(hash)
            class_number = possible_audio_classes[object_class]
        except:
            class_number = default_class
        sample_cache[hash] = class_number
        return class_number
    else:
        return sample_cache[hash]


# 'title': title,
# 'artist': artist,
# 'hash': rawhash,
# 'rawtext': rawtext,
# 'headers': headers,
# 'playables': playables,
# 'nonplayables': nonplayables

# For debug purpose, give a number index instead of whole array for note location.
def translateInputNoteToOutputNote(note):
    result = 0
    for i in reversed(range(8)):
        if note[2 * i]:
            result = result * 2 + 1
        else:
            result = result * 2
    return result


# Match the note to a potential note signature.
def getBeatAlignment(beat):
    epsilon = 0.000001
    for alignment in possible_alignment:
        if abs(int(beat * alignment) - beat * alignment) < epsilon:
            return alignment
    return 0

    # Input features - legacy
    # 0+16: On/Off for each columns, 8x2
    # 16+2: delta time to previous note, next note
    # 18+2: beat to previous note, next note
    # 20+4: which 16th note this note is padding to
    # 24+n: audio classification

# Standalone helper function to give some insight for the input data.
def analysisData(query={}, algorithm="osu"):
    dbInterface = MongoDBInterface()
    difficultyCalculator = DifficultyCalculator(algorithm)
    dbInterface.open()
    all_p_np_ratio = []
    all_p_lens = []
    all_diff = []
    counter = 0
    for doc in dbInterface.getChartDocument(query):
        # prepare chart-based data
        title = doc['headers']['title']
        bpm = getOrNone(doc['headers'], 'bpm')
        playables = doc['playables']
        nonplayables = doc['nonplayables']
        len_p = len(playables)
        if len_p < 50: # Skip bad Chart that contains less than 50 notes. Even the easiest IIDX chart has far more than this.
            continue
        counter = counter + 1
        difficulty = difficultyCalculator.calculate(doc)
        all_diff.append(difficulty)
        len_np = len(nonplayables)
        try:
            ratio = len_p / (len_p + len_np)
        except:
            ratio = 0
        all_p_np_ratio.append(ratio)
        all_p_lens.append(len_p)
        xth = []
        for playables in doc['playables']:
            beat = playables['data']['beat']
            beat_on_xth = getBeatAlignment(beat)
            xth.append(beat_on_xth)
        print("C#:%d\tDiff(%f)\t%s\t%s\t%s\t%s\t" % (counter-1,difficulty, len_p, len_np, bpm, title))
    density = stats.gaussian_kde(all_diff)
    n, x, _ = plt.hist(all_diff, 30, normed=1, facecolor='gray', alpha=0.75)
    plt.plot(x, density(x), linewidth=3, color="black")
    plt.xlabel('Difficulty')
    plt.ylabel('Probability')
    plt.title("Difficulty (" + algorithm + " algorithm)")
    plt.grid(True)
    plt.show()
    density = stats.gaussian_kde(all_p_np_ratio)
    n, x, _ = plt.hist(all_p_np_ratio, 30, normed=1, facecolor='gray', alpha=0.75)
    plt.plot(x, density(x), linewidth=3, color="black")
    plt.xlabel('Playable objects % in all objects')
    plt.ylabel('Probability')
    plt.title("Playable %")
    plt.grid(True)
    plt.show()
    density = stats.gaussian_kde(all_p_lens)
    n, x, _ = plt.hist(all_p_lens, 30, normed=1, facecolor='gray', alpha=0.75)
    plt.plot(x, density(x), linewidth=3, color="black")
    plt.xlabel('Playable objects count')
    plt.ylabel('Probability')
    plt.title("Number of playable objects")
    plt.grid(True)
    plt.show()

# LEGACY - prepare learning files without lookbacks.
# Possible Criteria:
# outfile : The output file for the generated pytorch data file.
# include_nonplayables : Unused keysounds in background are put into the data as all columns "off".
# combine_notes_at_the_same_time : Notes at the same time are combined to a single note, rounded by 1/x th (100 = .01s, 1000 = .001s, etc.)
# length_cutoff: If charts have less than X notes than the chart is discarded.
# create_unroll: Create data of length X.
def prepareLearningFilesFromDB(criteria={}):
    # make this accept some parameters
    dim_note_data = 25

    if getOrNone(criteria, 'combine_notes_at_the_same_time'):
        time_factor = criteria['combine_notes_at_the_same_time']
    else:
        time_factor = 1

    dbInterface = MongoDBInterface()
    dbInterface.open()
    notes = []
    all_data = []
    query_criteria = {}
    total = 0
    for doc in tqdm(dbInterface.getChartDocument(query_criteria)):
        all_result_for_chart = []
        all_result_for_chart.append([0] * dim_note_data)  # "start" token
        notes = {}
        all_notes = doc['playables']
        if getOrNone(criteria, 'include_nonplayables'):
            all_notes = all_notes + doc['nonplayables']
        print(len(all_notes))
        for playables in doc['playables']:
            if time_factor != 1:
                time = int(playables['time'] * time_factor)
            else:
                time = playables['time']
            beat = playables['data']['beat']
            beat_alignment = round((beat * 4)) % 4
            column = translateColumnNameToNumber(playables['data']['column'])
            if time not in notes:
                notes[time] = {'raw': [playables], 'beat': beat, 'align': beat_alignment, 'column': [False] * 8}
            else:
                notes[time]['raw'].append(playables)
            notes[time]['column'][column] = True
        sorted_notes_keys = sorted(notes)
        for index, key in enumerate(sorted_notes_keys):
            if index == 0:
                last_time = 0
                last_beat = 0
            else:
                last_time = sorted_notes_keys[index - 1]
                last_beat = notes[sorted_notes_keys[index - 1]]['beat']
            if index == len(sorted_notes_keys) - 1:
                next_time = 0
                next_beat = 0
            else:
                next_time = sorted_notes_keys[index + 1]
                next_beat = notes[sorted_notes_keys[index + 1]]['beat']
            this_time = key
            this_beat = notes[key]['beat']
            delta_time_prev = this_time - last_time
            delta_time_next = next_time - this_time
            delta_beat_prev = this_beat - last_beat
            delta_beat_next = next_beat - this_beat
            result = [0] * dim_note_data
            columns = notes[key]['column']
            for i in range(8):
                if columns[i]:
                    result[i * 2] = 1
                else:
                    result[i * 2 + 1] = 1
            result[0x10] = delta_time_prev
            result[0x10 + 0x1] = delta_time_next  # Next dt
            result[0x12] = delta_time_prev
            result[0x12 + 0x1] = delta_beat_next  # Next db
            beat_alignment = notes[key]['align']
            for i in range(4):
                if beat_alignment == i:
                    result[0x14 + i] = 1
                else:
                    result[0x14 + i] = 0
            all_result_for_chart.append(result)
        all_result_for_chart.append([0] * dim_note_data)
        if getOrNone(criteria, 'length_cutoff'):
            if len(all_result_for_chart) < getOrNone(criteria, 'length_cutoff'):
                continue
        reshaped = np.array(all_result_for_chart).reshape(-1, dim_note_data)
        reshaped = pad_array(reshaped)
        all_data.append(reshaped)
    if getOrNone(criteria, 'create_unroll'):
        all_data = np.concatenate(tuple(all_data), axis=0)
    else:
        all_data = np.array(all_data)
    print("All data:" + str(all_data.shape))
    print("All data0:" + str(all_data[1].shape))
    if getOrNone(criteria, 'outfile'):
        torch.save(all_data, open(criteria['outfile'], 'wb'), pickle_protocol=4)
    else:
        pass
        # torch.save(all_data, open('traindata.pt'),'wb')


# LEGACY - prepare features including difficulty.
def preparePlayableClassificationFile(criteria={}, algorithm="osu"):
    # make this accept some parameters
    dim_note_data = 32
    diffc = DifficultyCalculator(algorithm)
    dbInterface = MongoDBInterface()
    dbInterface.open()
    notes = []
    all_data = []
    query_criteria = {}
    # sample_cache = {}

    for doc in tqdm(dbInterface.getChartDocument(query_criteria)):
        # prepare chart-based data

        diff = diffc.calculate(doc)
        all_result_for_chart = []
        start_token = [0] * dim_note_data
        start_token[0] = diff
        all_result_for_chart.append([0] * dim_note_data)  # "start" token
        notes = {}
        playables = doc['playables']
        nonplayables = doc['nonplayables']
        if len(playables) + len(nonplayables) < 32:
            continue
        objects_without_class = 0
        objects_total = len(playables) + len(nonplayables)
        for object in playables:
            hash = object['hash']
            class_number = getSampleClass(hash)
            result = [0] * dim_note_data
            result[0] = diff
            time = object['time'] / 120.0
            result[1] = time
            result[2] = 1
            result[3] = 0
            result[4 + class_number] = 1
            all_result_for_chart.append(result)
        for object in nonplayables:
            hash = object['hash']
            class_number = getSampleClass(hash)
            result = [0] * dim_note_data
            result[0] = diff
            time = object['time'] / 120.0
            result[1] = time
            result[2] = 0
            result[3] = 1
            result[4 + class_number] = 1
            all_result_for_chart.append(result)

        def timeOfResult(arr):
            return arr[1]

        all_result_for_chart.sort(key=timeOfResult)
        all_result_for_chart.append([0] * dim_note_data)
        reshaped = np.array(all_result_for_chart).reshape(-1, dim_note_data)
        reshaped = pad_array(reshaped, PAD_LENGTH_PLAYABLE_CLASSIFICATION)
        all_data.append(reshaped)
    all_data = np.array(all_data)
    print("All data:" + str(all_data.shape))
    print("All data0:" + str(all_data[1].shape))
    if getOrNone(criteria, 'outfile'):
        torch.save(all_data, open(criteria['outfile'], 'wb'), pickle_protocol=4)
    else:
        pass
        # torch.save(all_data, open('traindata.pt'),'wb')

# Frame are the minimal unit used in total_beats.
# Notes can be coming in smaller intervals, it's just the training model ignores them.
def beat_to_frame(beat):
    return int(beat * 16)

# Prepare information for a note.
def generateCommonInfo(obj):
    info = {}
    hash = obj['hash']
    class_number = getSampleClass(hash)
    time = obj['time']
    beat = obj['data']['beat']
    beat -= int(beat)
    beat = beat_to_frame(beat)  # cutoff is at 16th
    return {'class': class_number, 'beat': beat, 'time': time}

# Normalize functions.
# Todo: unify them with copies in data_utils.py
def normalize(v):
    norm = np.sum(v)  # np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# normalize + weight added in favor of more data pts.
def normalize2(v):
    norm = np.sum(v)  # np.linalg.norm(v)
    if norm == 0:
        return v
    result = list(v)
    for index, elem in enumerate(v):
        result[index] = np.log(v[index] + 1) * v[index] / norm
    return result

def normalize3(v):
    max_index = -1
    max_value = -1
    max2_index = -1
    max2_value = -1
    for key in v:
        value = v[key][0] #playable
        if value > max_value:
            max2_index = max_index
            max2_value = max_value
            max_value = value
            max_index = key
        elif value > max2_value:
            max2_index = key
            max2_value = value
    result = {}
    for key in v:
        if max_index == key:
            result[key] = [1,0]
        elif max2_index == key:
            result[key] = [0,1]
        else:
            result[key] = [0,0]
    return result

# Generates an aggregated result.
# In the result, every column is the sum of `start` to `end`.
# Normalization also happen after this aggregation.
def aggregateBeatsToMeasures(arr, start, end):
    if start < 0:
        start = 0
    if end > len(arr):
        end = len(arr)
    result_obj = {}
    for idx in range(start, end):
        for key in arr[idx]:
            if key not in result_obj:
                result_obj[key] = arr[idx][key]
            else:
                result_obj[key] = np.add(arr[idx][key], result_obj[key])
    for key in result_obj:
        if normalize_method == 2:
            result_obj[key] = list(normalize2(result_obj[key]))
        elif normalize_method == 1:
            result_obj[key] = list(normalize(result_obj[key]))
        elif normalize_method == 3:
            pass # do nothing here, we need to do something different
        else:
            print("Unknown normalize method specified.")
            exit()
    if normalize_method == 3:
        result_obj = normalize3(result_obj)
    return result_obj

# Helper function to add a note to a given beat. Much like a hash table.
def addNote(target, beat, note):
    beat_int = int(beat)
    if beat_int in target:
        target[beat_int].append(note)
    else:
        target[beat_int] = [note]

# Need to use another function of same functionality
def column_to_number(col):
    if col == 'SC':
        return 30
    try:
        return int(col)
    except:
        print("Exception in column_to_number: Using SC(30) as output.")
        return 30

# Legacy function.
# def aggregateLookbackData(doc):
#     diffc = DifficultyCalculator("osu")
#     playable_counter = 0
#     diff = diffc.calculate(doc)
#     bpm = doc['headers']['bpm']
#     start_token = [0] * dim_note_data
#     start_token[0] = diff
#     # all_result_for_chart.append([0] * dim_note_data) # "start" token
#     notes = {}
#     playables = doc['playables']
#     nonplayables = doc['nonplayables']
#     if len(playables) < 50:
#         return []
#     max_beat = 0
#     note_list = {}
#     bg_list = {}
#     for obj in playables:
#         beat = beat_to_frame(obj['data']['beat'])
#         if max_beat < beat:
#             max_beat = beat
#         addNote(note_list, beat, obj)
#     for obj in nonplayables:
#         beat = beat_to_frame(obj['data']['beat'])
#         if max_beat < beat:
#             max_beat = beat
#         addNote(bg_list, beat, obj)
#     aggregated_data = {}
#     for beat_count in range(0, max_beat):
#         aggregated_data[beat_count] = []
#         for obj in note_list.get(beat_count, []):
#             info = generateCommonInfo(obj)
#             info['total_beat'] = beat_count
#             info['playable'] = True
#             info['column'] = column_to_number(obj['data']['column'])
#             aggregated_data[beat_count].append(info)
#         for obj in bg_list.get(beat_count, []):
#             info = generateCommonInfo(obj)
#             info['total_beat'] = beat_count
#             info['playable'] = False
#             info['column'] = 90
#             aggregated_data[beat_count].append(info)
#     flattened_data = []
#
#     for beat_count in range(0, max_beat):
#         this_beat = {}
#         for item in aggregated_data[beat_count]:
#             item_class = item['class']
#             playable = item['playable']
#             if item_class not in this_beat:
#                 this_beat[item_class] = [0, 0]
#             if playable:
#                 this_beat[item_class][0] += 1
#
#             else:
#                 this_beat[item_class][1] += 1
#         flattened_data.append(this_beat)
#
#     flattened_aggregated_data = []
#     for beat_count in range(0, max_beat):
#         if len(aggregated_data[beat_count]) < 1:
#             continue
#         this_note_data = {0: aggregated_data[beat_count]}
#         for factors in look_back:
#             this_note_data[factors] = aggregateBeatsToMeasures(flattened_data, beat_count - factors, beat_count)
#         flattened_aggregated_data.append(this_note_data)
#     this_song = []
#     for raw_data in flattened_aggregated_data:
#         for element in raw_data[0]:  # separate entry for every single object
#             this_note_data = np.zeros(dim_note_data)
#
#             if not do_per_note_diff:
#                 this_note_data[0] = diff
#             else:
#                 this_note_data[0] = diffc.getOsuNoteStrain(element['time'])
#             this_note_data[1] = element['beat']
#             if element['playable'] is True:
#                 this_note_data[2] = 1
#             else:
#                 this_note_data[3] = 1
#             if element['playable'] is True:
#                 playable_counter += 1
#             this_note_data[4 + element['class']] = 1
#             if getOrNone(criteria, 'extra_info'):
#                 this_note_data[31] = int(element['total_beat']) + 0.01 * element['column']
#             index = 32
#             for factors in look_back:
#                 for item in raw_data[factors]:
#                     true_idx = index + item * 2
#                     this_note_data[true_idx] = raw_data[factors][item][0]
#                     this_note_data[true_idx + 1] = raw_data[factors][item][1]
#                 index += total_classes * 2
#             this_song.append(this_note_data)
#     counter_this_song = 0
#     for elem in this_song:
#         if elem[2] == 1:
#             counter_this_song += 1
#     return this_song

# Function below generates the feature (style) file, both with difficulty and lookback features.
# Feature vectors are built as follows:
# Dim - Function
# 0 : Difficulty from the difficulty curve (if osu!mania method), else a single value throughout
# 1 : Which 64th beat it belongs to. If it's a quarter note the value is 0. Otherwise each 64th incremental the value increases by 1.
# 2 : 1 if the note is playable, 0 otherwise
# 3 : 1 if the note is unplayable, 0 otherwise
# 4 - 30: 1 if it belongs to class (value - 4), 0 otherwise
# 31 : int(time in frames) + 0.01 * column location in 0 for non-playable, [1,7] for normal notes and [30] for scratch.
# 32 + 54 * x: lookback information.
# Lookback information structure:
# For each information piece, all notes in [time in frames - lookback value,time in frames] have this calculated:
# Dim - Function
# 2x : 1 if this note is in class x and is playable, 0  otherwise
# 2x+1: 1 if this note is in class x and is non-playable, 0 otherwise
# After this all values are normalized based on normalization method selected.
# Default lookback values used start at half measure and doubles until it hits 8 measures.
def preparePlayableWithLookbackFile(criteria={}, algorithm="osu"):
    # make this accept some parameters
    dim_note_data = 304

    # First 8: Chart specific params
    # Next 32: Audio features

    diffc = DifficultyCalculator(algorithm)

    dbInterface = MongoDBInterface()
    dbInterface.open()
    notes = []
    all_data = []
    query_criteria = {}

    # sample_cache = {}
    all_max_beats = []
    total = 0
    total_cap = getOrNone(criteria, 'max_data')
    do_per_note_diff = getOrNone(criteria, 'per_note_diff_osu')
    if total_cap is None:
        print("Creating dataset for all charts in the database. This will take some time...")
        total_cap = 999999999
    else:
        print("Stop after %d charts processed." % total_cap)
    per_chart_all_data = []
    for doc in tqdm(dbInterface.getChartDocument(query_criteria)):
        playable_counter = 0
        total += 1
        if total > total_cap:
            break

        # if total < 425:
        #     continue
        # prepare chart-based data

        diff = diffc.calculate(doc)
        #print("Diff:%f" % diff)
        # if diff > 8:
        #     print(diff)
        # continue
        bpm = doc['headers']['bpm']
        # all_result_for_chart = []

        start_token = [0] * dim_note_data
        start_token[0] = diff
        # all_result_for_chart.append([0] * dim_note_data) # "start" token
        notes = {}
        playables = doc['playables']
        nonplayables = doc['nonplayables']
        if len(playables) < 50:
            continue
        max_beat = 0
        note_list = {}
        bg_list = {}
        #print("Playables: %d Non-playables: %d" % (len(playables), len(nonplayables)))
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
                info = generateCommonInfo(obj)
                info['total_beat'] = beat_count
                info['playable'] = True
                info['column'] = column_to_number(obj['data']['column'])
                aggregated_data[beat_count].append(info)
            for obj in bg_list.get(beat_count, []):
                info = generateCommonInfo(obj)
                info['total_beat'] = beat_count
                info['playable'] = False
                info['column'] = 90
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
                # print(this_note_data)
            flattened_aggregated_data.append(this_note_data)
        this_song = []
        for raw_data in flattened_aggregated_data:
            for element in raw_data[0]:  # separate entry for every single object
                this_note_data = np.zeros(dim_note_data)

                if not do_per_note_diff:
                    this_note_data[0] = diff
                else:
                    this_note_data[0] = diffc.getOsuNoteStrain(element['time'])
                    # print("!")
                    # print("Per note diff:%f"%this_note_data[0])
                this_note_data[1] = element['beat']
                if element['playable'] is True:
                    this_note_data[2] = 1
                else:
                    this_note_data[3] = 1
                # this_note_data[2] = element['playable'] is True
                # this_note_data[3] = element['playable'] is False
                if element['playable'] is True:
                    playable_counter += 1
                this_note_data[4 + element['class']] = 1
                if getOrNone(criteria, 'extra_info'):
                    this_note_data[31] = int(element['total_beat']) + 0.01 * element['column']
                    # this_note_data[30] = element['column']
                index = 32
                for factors in look_back:
                    for item in raw_data[factors]:
                        true_idx = index + item * 2
                        this_note_data[true_idx] = raw_data[factors][item][0]
                        this_note_data[true_idx + 1] = raw_data[factors][item][1]
                    index += total_classes * 2
                #print(this_note_data)
                this_song.append(this_note_data)
                all_data.append(this_note_data)
                # print(this_note_data)
        # print(np.array(this_song).shape)
        counter_this_song = 0
        for elem in this_song:
            if elem[2] == 1:
                counter_this_song += 1
        #print("playable in this_song:%d" % counter_this_song)
        #print("Playable counted:%d" % playable_counter)
        #print("length of playable[] is %d." % len(doc['playables']))
        per_chart_all_data.append(this_song)
        # print("Current all data length:%d"%(all_data.__len__()))
        #print("Current per-chart data size:%d", per_chart_all_data.__len__())
        #print("per-chart 0:%d" % len(per_chart_all_data[0]))
    all_data = np.array(all_data)
    per_chart_all_data = np.array(per_chart_all_data)
    #print("All data:" + str(all_data.shape))
    # print("Per chart all data:%d"%str(per_chart_all_data.shape))
    if getOrNone(criteria, 'outfile'):
        torch.save(all_data, open(criteria['outfile'], 'wb'), pickle_protocol=4)
    else:
        pass
        # torch.save(all_data, open(criteria['outfile'], 'wb'), pickle_protocol=4)

    if getOrNone(criteria, 'outfile_per_chart'):
        torch.save(per_chart_all_data, open(criteria['outfile_per_chart'], 'wb'), pickle_protocol=4)
    else:
        pass
    print("Done.")
