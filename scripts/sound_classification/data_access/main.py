# Stand alone entry point of the data access module.

from BMStoDB import BMSDBAdapter
from tqdm import tqdm
import os
import sys

try:
    input_path = sys.argv[1]
    print("Parsing all BMS files in "+str(input_path))
except:
    print("No path specified.")
    exit()

#'/home/zhiyulin/Projects/rawdata/bof2011'

ad = BMSDBAdapter()
#ad.dumpWholeDirectory('/home/zhiyulin/Projects/rawdata/bof2011/ГLГ`В╠Ч╓/fylaki')
for path,dirs,files in tqdm(os.walk(input_path)):
    print("Working on: "+str(path))
    ad.dumpWholeDirectory(path)
print("Finished!")