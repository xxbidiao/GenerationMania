# data_utils.py
# A bunch of helper functions.
import numpy as np
import torch
from bisect import *
import statistics
import os,fnmatch

# These functions, by default, connects two sequences in the long side.
# [20,x]+[20,x] = [40,x] not [20,2x]
# Connect two seqs (numpy style)
def pad_sequences(seq1,seq2):
    return np.concatenate([seq1,seq2],axis=1)

# Connect two seqs (torch style)
def pad_sequences_torch(seq1,seq2,dim=2):
    result = torch.cat((seq1,seq2),dim)
    return result

def find_ge(a, x):
    #Find leftmost item greater than or equal to x, if not find return len of array
    i = bisect_left(a, x)
    return i

# Default lookback length. 64 = 1 measure.
look_back = [32, 64, 128, 256, 512]

total_classes = 27

# Used in generate feedforward results
def aggregateDataAndNormalize(arr,start,end,colstart,colend):
    result = np.zeros(colend - colstart + 1)
    for i in range(start,end):
        j_idx = 0
        for j in range(colstart,colend):
            result[j_idx] += arr[i,j]
    for i in range(colstart,colend,2):
        total = result[i] + result[i+1]
        if total == 0:
            continue
        result[i] = result[i] / total
        result[i+1] = result[i+1] / total
    return result

# Aggregate lookback results from `arr`, from time window `start` to `end`,
# by using method `normalize_method`.
def aggregate_lookback(arr,start,end,normalize_method=2):
    result = [0] * (total_classes*2)
    playable_offset = 2
    audio_class_offset = 4
    beat_arr = []
    non_zero_met = False
    for item in arr:
        total_beat = int(item[31]) # only take the total beat information, discarding column info
        if total_beat > 0:
            non_zero_met = True
        if non_zero_met and total_beat > 0:
            beat_arr.append(total_beat)
        else: # If we met one non-zero before, this zero means End Of Chart instead.
            beat_arr.append(1999999999)
    starting_point = find_ge(beat_arr,start)
    # First, count occurences of playables and nonplayables.
    for idx in range(starting_point,len(arr)):
        total_beat = int(arr[idx][31])
        if total_beat >= end:
            break
        if arr[idx][playable_offset] == 1: #playable
            offset = 0
        else:
            offset = 1
        for class_idx in range(total_classes):
            if arr[idx][audio_class_offset + class_idx] == 1:
                result[class_idx * 2 + offset] += 1
    # Then normalize them.
    if normalize_method == 3:
        result = normalize3_array(result)
    for i in range(total_classes):
        if normalize_method == 1:
            sub_result = normalize([result[i*2],result[i*2+1]])
            result[i * 2] = sub_result[0]
            result[i * 2 + 1] = sub_result[1]
        elif normalize_method == 2:
            sub_result = normalize2([result[i * 2], result[i * 2 + 1]])
            result[i * 2] = sub_result[0]
            result[i * 2 + 1] = sub_result[1]
        elif normalize_method == 3:
            pass # Handled already
        else:
            print("Error:Unknown normalization method specified.")
            exit()
    return result


# Generates full lookback profile for note at `arr`[`index`], using default lookback length specified.
def aggregate_lookback_all(arr,index,normalize_method = 2):
    real_index = int(arr[index][31])
    total_result = {}
    for lb_value in look_back:
        total_result[lb_value] = aggregate_lookback(arr,real_index-lb_value,real_index,normalize_method=normalize_method)
    return total_result

# Concatenate the original features array with lookback result array.
def recombine_feedforward_results(arr_orig,lookback_result,start=32):
    result = arr_orig
    index = start
    for factors in look_back:
        for idx in range(len(lookback_result[factors])):
            result[index] = lookback_result[factors][idx]
            index += 1
    return result

# a normalization method that also takes in the total number.
# value = value / nsum(all values) * log(value)
def normalize2(v):
    norm = np.sum(v)#np.linalg.norm(v)
    if norm == 0:
       return v
    result = v
    for index,elem in enumerate(v):
        result[index] = np.log(v[index]+1) * v[index] / norm
    return result

# the most standard normalization method.
# value = value / sum(all values)
def normalize(v):
    norm = np.sum(v)#np.linalg.norm(v)
    if norm == 0:
       return v
    result = v
    for index,elem in enumerate(v):
        result[index] = v[index] / norm
    return result

# A "normalization" method that actually takes the top 2.
# makes a one-hot encoding what is the most prevalent instrument class.
def normalize3_array(v):
    max_index = -1
    max_value = -1
    max2_index = -1
    max2_value = -1
    for index,value in enumerate(v):
        if index%2==1:
            continue #skip all nonplayables
        if value > max_value:
            max2_value = max_value
            max2_index = max_index
            max_value = value
            max_index = index
        elif value > max2_value:
            max2_index = index
            max2_value = value
    result = [0]*len(v)
    if max_index >= 0:
        result[max_index] = 1
    if max2_index >= 0:
        result[max2_index+1] = 1
        #pass
    return result

# Get the average of `arr`.
def avg(arr):
    try:
        return sum(arr) / len(arr)
    except:
        print("Something wrong happened in avg()")
        return -1

# Get the standard deviation of `arr`.
def std(arr):
    try:
        return statistics.stdev(arr)
    except:
        print("Something wrong happened in std()")
        return -1

#https://stackoverflow.com/questions/17282887/getting-files-with-same-name-irrespective-of-their-extension
def get_all_files(path, pattern):
    datafiles = []
    for root,dirs,files in os.walk(path):
        for file in fnmatch.filter(files, pattern):
            pathname = os.path.join(root, file)
            filesize = os.stat(pathname).st_size
            datafiles.append(file)
    return datafiles

# Turns column names (from bms-js output) to numbers.
# Scratchs are counted as 0.
def translateColumnNameToNumber(col):
    try:
        return int(col)
    except ValueError:
        if col == 'SC':
            return 0
        else:
            print("Unknown Column Name found: " + str(col))
            return None

