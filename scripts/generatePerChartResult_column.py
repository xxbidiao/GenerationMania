# generatePerChartResult_column.py
# Nearly the same as the one without column in filename, this is for evaluating on columns.
# TODO: merge these two files.

import numpy as np
import matplotlib.pyplot as plt
import torch
from .data_utils import *
from .DBtoLearn import aggregateBeatsToMeasures
from .model_feedforward import eval_model
from .model_ff_cols import eval_model as eval_model_cols
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

isCuda = False
if isCuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

pad_correct_history_data_ratio = 0
random_positive_chance = 1
random_evaluation_result = False
using_state = config.training_state_feedforward_columns
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

def get_col(item):
    #print("Get col got called.")
    col = np.rint((item[31]-np.rint(item[31]))*100)
    #print("ORIG:%f,col:%d"%(item[31],col))
    if col == 30: #scratch
        return 0 # not 8
    if 8 > col > 0:
        return col
    return 8 # not -1
        # if col not in range(1, 8):
        #     print("OOB column:%d"%col)

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

def eval_test():
    global data
    #print("Loading data from disk...")
    if data is None:
        print("Loading data from disk...")
        data = torch.load(data_file_name)
        print("Loaded!")
    #print("Loaded!")
                    #test code
    data0 = data[0]
    arr0 = np.zeros(np.array(data0).shape)
    data0_index = 0
    total = 0
    #print(len(data[0]))
    for note in data[13]:
        total += 1
        # if total  > 5:
        #     exit()
        comparing_base = [0]*304
        comparing_to = note
        for idx in range(32):
            comparing_base[idx] = comparing_to[idx]
        arr0[total-1] = comparing_base
        raw_lookback = aggregate_lookback_all(arr0,data0_index)
        #print(raw_lookback)
        comparing_base = recombine_feedforward_results(comparing_base,raw_lookback)
        # print("@@@")
        # print(comparing_base)
        # print("---")
        # print(comparing_to)
        for idx in range(len(comparing_base)):
            if comparing_base[idx] != comparing_to[idx]:
                print("note %d index %d unmatch, value %f vs correct %f"%(total,idx,comparing_base[idx],comparing_to[idx]))
                #print("index %d unmatch++, value %f vs correct %f" % (idx-1, comparing_base[idx-1], comparing_to[idx-1]))

        arr0[total-1] = comparing_base
        data0_index+=1
    print("Validation ok")

def change_repr_diff(repr,diff):
    result = copy.deepcopy(repr)
    result[0] = diff
    return result

def classof(repr):
    result = eval_model(repr[:]).data.numpy()
    result_max_index = np.argmax(result)
    return result_max_index

# find the boundary of playable/nonplayable for a given data point
def eval_difficulty(repr,target,delta = 0.1):
    #print("Start eval_difficulty")
    if classof(change_repr_diff(repr,0)) == 0:
        #print("Diff 0 is playable for this note")
        return -0.05 #edge case: even diff 0 gives a playable note
    diff = repr[0]
    #print("Diff:%f"%diff)
    if diff < 0.01:
        #print("Diff < 0.01")
        return 0
    result_max_index = classof(repr)
    if result_max_index == 0: #playable
        lower_bound = 0
        upper_bound = diff
    else:
        lower_bound = diff
        upper_bound = diff * 2
        while classof(change_repr_diff(repr,upper_bound)) == 1:
            #print("Oh %f"%upper_bound)
            upper_bound *= 2
            if upper_bound > 50:
                #print("Need inf difficulty to make this playable")
                return 50.5
    #print("L/U bound:%f vs %f"%(lower_bound,upper_bound))
    while upper_bound > lower_bound + delta:
        this_try = (upper_bound + lower_bound) / 2.0
        if classof(change_repr_diff(repr,this_try)) == 0:
            upper_bound = this_try
        else:
            lower_bound = this_try
    #print("L/U bound:%f vs %f" % (lower_bound, upper_bound))
    return (lower_bound + upper_bound)/2

def evaluate(chart_num = -1,diff_modifier = 1):
    #print("Start: pad_history_ratio = %f"%pad_correct_history_data_ratio)
    global data
    #print("Loading data from disk...")
    if data is None:
        print("Loading data from disk...")
        data = torch.load(data_file_name)
        print("Loaded!")
    #print("Loaded!")
    if random_evaluation_result:
        print("Randomizing predicted results with %f chance of positive."%random_positive_chance)

    length = data.shape[0]

    #This is actually test_size. todo: rename them.
    verif_size_start = length - 1 * int(length * 1.0 / 10)
    verif_size_end = length# - int(length * 1.0 / 10)
    #print("Start,end:%d %d"%(verif_size_start,verif_size_end))
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
        #print("Processing chart #%d"%chart_index)
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
        # po_counter = 0
        # for elem in chart_data:
        #     if elem[2] == 1:
        #         po_counter += 1
        #print("po_counter = %d"%po_counter)
        #print(len(chart_data))
        pad_correct_history_data = pad_correct_history_data_ratio * len(chart_data)
        all_single_note_diff = []
        for note_index,note in enumerate(chart_data):
            #print("Note %d"%note_index)
            # print("comparing-array")
            # print(comparing_array)
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

            # print(note[0])
            # print(comparing_base[0])

            #print(comparing_base)
            #print(comparing_to)
            if not random_evaluation_result:
                actual_diff = comparing_base[0]
                # print("Actual_diff:%f"%comparing_base[0])
                comparing_base[2] = 0  # playable flags
                comparing_base[3] = 0  # removing them,though the evaluator don't take them too
                comparing_array[note_index] = comparing_base
                if do_aggregate:
                    raw_lookback = aggregate_lookback_all(comparing_array, note_index,1)
                    # print(raw_lookback)
                    comparing_base = recombine_feedforward_results(comparing_base, raw_lookback)
                result = eval_model_cols(anti_tr(comparing_base),using_state).cpu().data.numpy()
            else:

                if random.random() < random_positive_chance:
                    result = [1,0]
                else:
                    result = [0,1]
            result_max_index = np.argmax(result)
            comparing_base[2+result_max_index] = 1
            correct_max_index = get_col(comparing_to)#np.argmax(comparing_to[2:4])
            #print("Predicted vs. correct: %s %s"%(result_max_index,correct_max_index))
            #print("Comparing-to:%s"%comparing_to[2:4])
            if comparing_to[2] == comparing_to[3]:
                print("Something wrong is in the data.")
            #expected_diff = eval_difficulty(anti_tr(comparing_base),correct_max_index)
            expected_diff = 0
            note_expected_diffs.append(expected_diff)

            # if np.array_equal(comparing_to,comparing_base):
            #     print("Same!")
            if note_index >= pad_correct_history_data:
                #print("Pad base")
                comparing_array[note_index] = comparing_base
            else:
                #print("Pad original")
                comparing_array[note_index] = comparing_to
            # print(result)
            # if result[0]+result[1] > 1.2:
            #     print(result)

            period_start = 99999
            period_end = 99999
            #print("%d vs correct %d"%(result_max_index,correct_max_index))
            if correct_max_index != 8:
                if abs(correct_max_index - result_max_index) == 0:
                    tp += 1
                else:
                    fp += 1

            # if correct_max_index == 0:
            #     counter += 1
            #     if result_max_index == 0:
            #         tp += 1
            #         if np.abs(np.rint(result[0][0]) - get_col(comparing_to)) < 0.5:
            #             tp_hits += 1
            #         if(int(note[31]) in range(period_start,period_end)):
            #             corrects.append((result[0][0],int(note[31])))
            #     else:
            #         fn += 1
            #         if(int(note[31]) in range(period_start,period_end)):
            #             incorrects.append((result[0][0],int(note[31])))
            # else:
            #     if result_max_index == 0:
            #         fp += 1
            #         if(int(note[31]) in range(period_start,period_end)):
            #             incorrects.append((result[0][0],int(note[31])))
            #     else:
            #         tn += 1
            #         if(int(note[31]) in range(period_start,period_end)):
            #             corrects.append((result[0][0],int(note[31])))


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
        #str1+=("column hits:\t%f +/- %f\n" % (avg(tp_hits_all), std(tp_hits_all)))
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


    #     median_diff = np.median(note_expected_diffs)
    #     print("Median predicted diff:%f"%median_diff)
    #     # do it again, with difficulty substituted by median difficulty.
    #     for note_index,note in tqdm(enumerate(chart_data)):
    #         comparing_base = [0]*304
    #         comparing_to = note[:]
    #         for idx in range(32):
    #             comparing_base[idx] = comparing_to[idx]
    #         comparing_base[0] = median_diff
    #         actual_diff = comparing_base[0]
    #         comparing_base[2] = 0 #playable flags
    #         comparing_base[3] = 0 #removing them,though the evaluator don't take them too
    #         comparing_array[note_index] = comparing_base
    #         raw_lookback = aggregate_lookback_all(comparing_array, note_index)
    #         comparing_base = recombine_feedforward_results(comparing_base,raw_lookback)
    #
    #         result = eval_model(anti_tr(comparing_base)).data.numpy()
    #         result_max_index = np.argmax(result)
    #         correct_max_index = np.argmax(comparing_to[2:4])
    #         if comparing_to[2] == comparing_to[3]:
    #             print("Something wrong is in the data.")
    #         expected_diff = eval_difficulty(anti_tr(comparing_base),correct_max_index)
    #         note_expected_diffs.append(expected_diff)
    #         if expected_diff > 30:
    #             result_max_index = 1
    #         else:
    #             result_max_index = 0
    #         if correct_max_index == 0:
    #             counter += 1
    #             if result_max_index == 0:
    #                 tp += 1
    #             else:
    #                 fn += 1
    #         else:
    #             if result_max_index == 0:
    #                 fp += 1
    #             else:
    #                 tn += 1
    #
    #     median_diff = np.median(note_expected_diffs)
    #     print("tp=%d tn=%d fp=%d fn=%d" % (tp, tn, fp, fn))
    #     try:
    #         precision = 1.0 * tp / (tp + fp)
    #         recall = 1.0 * tp / (tp + fn)
    #
    #         f1_score = 2 / (1 / precision + 1 / recall)
    #         print("F1 score = %f" % f1_score)
    #         f1_2.append(f1_score)
    #     except:
    #         f1_2.append(0)
    #
    #     print("Median of expected diff = %f, while actual diff is %f."%(np.median(note_expected_diffs),actual_diff))
    f1_2 = [0]
    diff_f1_trans = [list(x) for x in zip(*diff_f1_pair)]

    plt.plot(diff_f1_trans[0], diff_f1_trans[1], 'bo')
    plt.xlabel("Difficulty (log scale)")
    plt.ylabel("Performance (F-score)")
    plt.show()
    # def avg(arr):
    #     try:
    #         return sum(arr)/len(arr)
    #     except:
    #         #print("Something wrong happened in avg()")
    #         return -1
    #print("Average F1 Score: %f / changed threshold:%f"%(avg(f1_1),avg(f1_2)))
    #return avg(f1_1)
    return str1

# results = {}
# for i in range(0,11):
#     pad_correct_history_data_ratio = 0.1 * i
#     F1 = evaluate()
#     results[pad_correct_history_data_ratio] = F1
#     print(results)



# for i in pad_ratio_arr:
#     pad_correct_history_data_ratio = i
#     result_str = evaluate()
#     setup_str = ("Setup: model=%s pad_hist=%f\n"%(using_state,pad_correct_history_data_ratio))
#     #print(setup_str)
#     print(setup_str+result_str)
#     results[i] = setup_str+result_str

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