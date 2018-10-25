import os
import json
import operator
import heapq
import time
import hashlib
import random
f = open('wordlist.txt', 'r').readlines()
w = open('output.txt', 'w')
d = {}
previous = time.time()
for root, dirs, files in os.walk("./"):
    for di in dirs:
        flip = random.randint(0, 0)  #switch to randint(0, 1) for coin flip
        if flip == 0:
            for fi in os.listdir(os.path.join(root, di)):
                current = time.time()
                if (current - previous > 10):
                    previous = current
                    w.write(json.dumps(d))
                temp = {}
                size = 0
                for word in f:
                    wd = word.strip()
                    if wd in fi:
                        print wd + fi
                        size += 1
                        if not wd in temp:
                            temp[wd] = 1
                        else:
                            temp[wd] += 1
                for key in temp:
                    if key not in d:
                        d[key] = float(100 * temp[key]) / float(size)
                    else:
                        d[key] += float(100 * temp[key]) / float(size)
w.write(json.dumps(d))
w.close()
stats = heapq.nlargest(30, d.iteritems(), key=operator.itemgetter(1))
print("30 most frequent items in order:")
for i in range(30):
    print(stats[i])