# comparison.py
# A standalone utility stub to test whether two torch serialized array are exactly the same; If not, printing out all differences.
import torch

file1 = "traindata/Playable-per-chart.pt"
file2 = "traindata/Playable-per-chart-ref.pt"

data1 = torch.load(file1)
data2 = torch.load(file2)

print(data1.shape)
print(data2.shape)

good = 0
bad = 0
for index,value in enumerate(data1):
    for index2,value2 in enumerate(value):
        # print(type(value2))
        # print(type(data2[index][index2]))
        if (value2 == data2[index][index2]).all():
            good += 1
            #print("OK for %d %d"%(index,index2))
        else:
            bad += 1
            #print("BAD DATA. %s vs %s"%(value2,data2[index][index2]))
            print("BAD DATA.%s"%(value2 == data2[index][index2]))
            #input("Press Enter to continue...")

print("Compare done. %d passes, %d fails."%(good,bad))
