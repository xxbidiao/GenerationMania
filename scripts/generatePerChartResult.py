# generatePerChartResult.py
# This file provides utilities for automatically evaluating the generated charts.

import numpy as np
import matplotlib.pyplot as plt
import torch
from .data_utils import *
from .DBtoLearn import aggregateBeatsToMeasures
from .model_feedforward import eval_model
import copy
from tqdm import tqdm
import random
import statistics
import json
import matplotlib.pyplot as plt
from .Config import config

use_history_in_training = True
use_diff_in_training = True
use_audio_features_in_training = True

# static switch for using cuda or not. Change this to your preference.
isCuda = False
if isCuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

pad_correct_history_data_ratio = 0
random_positive_chance = 1
random_evaluation_result = False
using_state = config.training_state_feedforward
data = None
#pad_ratio_arr = [0, 0.05, 0.1, 0.2, 0.4, 0.7, 1]
#pad_ratio_arr = [0]
pad_ratio_arr = [0]
#diff_modifier = [10,5,2,1,0.5,0.2,0.1]
diff_modifier = [1]
current_diff_modifier = 1
result_file_name = config.result_feedforward
do_aggregate = False
do_original = True
data_file_name = config.training_file_per_chart

# Extract column information from the dataset for ground truth notes.
def get_col(item):
    col = np.rint((item[31]-np.rint(item[31]))*100)
    if col == 30: #scratch
        return 8 # This is different from data_utils!
    if col > 0:
        return col
    return -1

# This function extract features from the data.
def anti_tr(arr):
    # removing history + audio classes
    # result = arr[:,0:2]
    # return result

    # # removing history
    # result = arr[0:32]
    # result[31] = 0
    # #print(result)
    # return np.delete(result,[2,3],axis=0)

    # full, only removing labels themselves
    arr2 = copy.deepcopy(arr)

    if not use_diff_in_training:
        arr2[0] = 0

    if not use_audio_features_in_training:
        for i in range(4,32):
            arr2[i] = 0

    if not use_history_in_training:
        for i in range(32, len(arr2)):
            arr2[i] = 0

    #removing time signatures.
    arr2[31] = 0


    #removing playable tags.
    return np.delete(arr2,[2,3],axis=0)

# Legacy debug stub for checking integrity of input features.
# def eval_test():
#     global data
#     #print("Loading data from disk...")
#     if data is None:
#         print("Loading data from disk...")
#         data = torch.load(data_file_name)
#         print("Loaded!")
#     data0 = data[0]
#     arr0 = np.zeros(np.array(data0).shape)
#     data0_index = 0
#     total = 0
#     #print(len(data[0]))
#     for note in data[13]:
#         total += 1
#         comparing_base = [0]*304
#         comparing_to = note
#         for idx in range(32):
#             comparing_base[idx] = comparing_to[idx]
#         arr0[total-1] = comparing_base
#         raw_lookback = aggregate_lookback_all(arr0,data0_index)
#         comparing_base = recombine_feedforward_results(comparing_base,raw_lookback)
#         for idx in range(len(comparing_base)):
#             if comparing_base[idx] != comparing_to[idx]:
#                 print("note %d index %d unmatch, value %f vs correct %f"%(total,idx,comparing_base[idx],comparing_to[idx]))
#         arr0[total-1] = comparing_base
#         data0_index+=1
#     print("Validation ok")

# Change all notes in the `repr` to have difficulty `diff`.
def change_repr_diff(repr,diff):
    result = copy.deepcopy(repr)
    result[0] = diff
    return result

# Is it playable/nonplayable for `repr`?
def classof(repr):
    result = eval_model(repr[:]).data.numpy()
    result_max_index = np.argmax(result)
    return result_max_index

# Experimental stub.
# find the boundary of playable/nonplayable for a given data point
def eval_difficulty(repr,target,delta = 0.1):
    if classof(change_repr_diff(repr,0)) == 0:
        # Difficulty 0 is playable for this note
        return -0.05 #edge case: even diff 0 gives a playable note
    diff = repr[0]
    if diff < 0.01:
        return 0
    result_max_index = classof(repr)
    if result_max_index == 0: #playable
        lower_bound = 0
        upper_bound = diff
    else:
        lower_bound = diff
        upper_bound = diff * 2
        while classof(change_repr_diff(repr,upper_bound)) == 1:
            upper_bound *= 2
            if upper_bound > 50:
                #Need inf difficulty to make this playable.
                return 50.5
    while upper_bound > lower_bound + delta:
        this_try = (upper_bound + lower_bound) / 2.0
        if classof(change_repr_diff(repr,this_try)) == 0:
            upper_bound = this_try
        else:
            lower_bound = this_try
    return (lower_bound + upper_bound)/2


def evaluate(chart_num = -1,diff_modifier = 1):
    # Pad some of the ground truth in history calculation if `pad_history_ratio` > 0.
    print("Start: pad_history_ratio = %f"%pad_correct_history_data_ratio)
    global data
    if data is None:
        print("Loading data from disk...")
        data = torch.load(data_file_name)
        print("Loaded!")
    if random_evaluation_result:
        print("Randomizing predicted results with %f chance of positive."%random_positive_chance)
    length = data.shape[0]

    #This is actually test_size. todo: rename them.
    verif_size_start = length - 1 * int(length * 1.0 / 10)
    verif_size_end = length# - int(length * 1.0 / 10)
    str1=""
    f1_1 = []
    precision_1 = []
    recall_1 = []
    tp_hits_all = []
    f1_2 = []
    diff_f1_pair = []
    for chart_index,chart_data in enumerate(data[verif_size_start:verif_size_end]):
        if chart_num >= 0:
            if chart_index > chart_num:
                break
            elif chart_index < chart_num:
                continue
        comparing_array = np.zeros(np.array(chart_data).shape)
        tp_hits = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        counter = 0
        note_expected_diffs = []
        corrects = []
        incorrects = []
        pad_correct_history_data = pad_correct_history_data_ratio * len(chart_data)
        all_single_note_diff = []
        for note_index,note in enumerate(chart_data):
            comparing_base = [0]*304
            comparing_to = note[:]
            if do_original:
                for idx in range(len(comparing_base)):
                    comparing_base[idx] = comparing_to[idx]
            else:
                for idx in range(32):
                    comparing_base[idx] = comparing_to[idx]
            diff_this_note = comparing_base[0]
            if len(all_single_note_diff) == 0:
                all_single_note_diff.append(diff_this_note)
            else:
                if all_single_note_diff[-1] != diff_this_note:
                    all_single_note_diff.append(diff_this_note)
            comparing_base[0] *= diff_modifier
            if not random_evaluation_result:
                actual_diff = comparing_base[0]
                comparing_base[2] = 0  # playable flags
                comparing_base[3] = 0  # removing them,though the evaluator don't take them too
                comparing_array[note_index] = comparing_base
                if do_aggregate: # self-aggregation
                    raw_lookback = aggregate_lookback_all(comparing_array, note_index,1)
                    comparing_base = recombine_feedforward_results(comparing_base, raw_lookback)
                result = eval_model(anti_tr(comparing_base),using_state).cpu().data.numpy()
            else:
                if random.random() < random_positive_chance:
                    result = [1,0]
                else:
                    result = [0,1]
            result_max_index = np.argmax(result)
            comparing_base[2+result_max_index] = 1
            correct_max_index = np.argmax(comparing_to[2:4])
            if comparing_to[2] == comparing_to[3]:
                print("Something wrong is in the data.")
            expected_diff = 0
            note_expected_diffs.append(expected_diff)
            if note_index >= pad_correct_history_data:
                #print("Pad base")
                comparing_array[note_index] = comparing_base
            else:
                #print("Pad original")
                comparing_array[note_index] = comparing_to
            # This can be used for one-off check: Is the generated note on or near the ground truth note?
            period_start = 99999
            period_end = 99999
            if correct_max_index == 0:
                counter += 1
                if result_max_index == 0:
                    tp += 1
                    # if np.abs(np.rint(result[0][0]) - get_col(comparing_to)) < 1.5:
                    #     tp_hits += 1
                    if(int(note[31]) in range(period_start,period_end)):
                        corrects.append((result[0][0],int(note[31])))
                else:
                    fn += 1
                    if(int(note[31]) in range(period_start,period_end)):
                        incorrects.append((result[0][0],int(note[31])))
            else:
                if result_max_index == 0:
                    fp += 1
                    if(int(note[31]) in range(period_start,period_end)):
                        incorrects.append((result[0][0],int(note[31])))
                else:
                    tn += 1
                    if(int(note[31]) in range(period_start,period_end)):
                        corrects.append((result[0][0],int(note[31])))
        all_single_note_diff.sort(reverse=True) # Sort from highest to lowest strain.
        weight = 1
        decay_weight = 0.9
        difficulty = 0
        for strain in all_single_note_diff:
            difficulty += weight * strain
            weight *= decay_weight
            if weight < 0.001:
                break
        try:
            precision = 1.0 * tp / (tp + fp)
            recall = 1.0 * tp / (tp + fn)
            f1_score = 2 / (1 / precision + 1 / recall)
            precision_1.append(precision)
            recall_1.append(recall)
            print("Chart #%d\tF1 = %f\t(tp=%d tn=%d fp=%d fn=%d)" % (chart_index+verif_size_start,f1_score,tp, tn, fp, fn))
            #print("F1 score = %f" % f1_score)
            #print("TP_COLUMN_HITS = %f"%(tp_hits/tp))
            tp_hits_all.append(tp_hits / tp)
            f1_1.append(f1_score)
            if np.log(difficulty) < 6.5:
                diff_f1_pair.append((np.log(difficulty*0.018),f1_score))
            else:
                print("Chart skipped. Diff = %f"%np.log(difficulty))
        except:
            print("Chart #%d\tF1 = %f\t(tp=%d tn=%d fp=%d fn=%d)" % (chart_index+verif_size_start, 0, tp, tn, fp, fn))
            f1_1.append(0)
            tp_hits_all.append(0)
            if np.log(difficulty) < 6.5:
                diff_f1_pair.append((np.log(difficulty*0.018),0))
            else:
                print("Chart skipped. Diff = %f"%np.log(difficulty))
            precision_1.append(0)
            recall_1.append(0)
        tp = fp = tn = fn = 0
        str1=("Running Summary:\n")
        setup_str = ("Setup: model=%s diff_mod=%f pad_hist=%f\n" % (using_state, diff_modifier,pad_correct_history_data_ratio))
        str1+=setup_str
        str1+=("F1:\t%f +/- %f\n"%(avg(f1_1),std(f1_1)))
        str1+=("precision:\t%f +/- %f\n" % (avg(precision_1), std(precision_1)))
        str1+=("recall:\t%f +/- %f\n" % (avg(recall_1), std(recall_1)))
        if chart_index % 10 == 5:
            print(str1)

        #plot correct/inc distribution
        # print(len(corrects))
        # cor_trans = [list(x) for x in zip(*corrects)]
        # incor_trans = [list(x) for x in zip(*incorrects)]
        # plt.plot(cor_trans[1], cor_trans[0], 'bo')
        # plt.plot(incor_trans[1], incor_trans[0], 'yo')
        #
        # plt.xlabel("Time in beat frames")
        # plt.ylabel("Playable Activation Level")
        # plt.show()
    f1_2 = [0]
    diff_f1_trans = [list(x) for x in zip(*diff_f1_pair)]
    plt.plot(diff_f1_trans[0], diff_f1_trans[1], 'bo')
    plt.xlabel("Difficulty (log scale)")
    plt.ylabel("Performance (F-score)")
    plt.show()
    return str1

# Batch evaluation stubs. Can be used to evaluate multiple setups.
def eval_all():
    results = {}
    for i in diff_modifier:
        result_str = evaluate(chart_num=-1,diff_modifier=i)
        setup_str = ("Setup: model=%s diff_mod=%f\n"%(using_state,i))
        #print(setup_str)
        print(setup_str+result_str)
        results[i] = setup_str+result_str
        print(json.dumps(results,indent=4))
    print("All results:")
    print(results)


# for i in pad_ratio_arr:
#     results[i] = []
# for chart_id in range(0,150):
#     for i in pad_ratio_arr:
#         pad_correct_history_data_ratio = i
#         F1 = evaluate(chart_num=chart_id)
#         if F1 > -0.01:
#             results[pad_correct_history_data_ratio].append(F1)
#         #print(results)
#     #print(results)
#     for key in results:
#         if chart_id == 0:
#             continue
#         print("h-ratio %f\t mean:%f stdev:%f"%(key,statistics.mean(results[key]),statistics.stdev(results[key])))
#     torch.save(results, open(result_file_name, 'wb'), pickle_protocol=4)