# Dataimportmain.py
# Interface of the data access module.

from .BMStoDB import BMSDBAdapter
from tqdm import tqdm
import os

def import_data(input_path):
    print("Parsing all BMS files in " + str(input_path))
    ad = BMSDBAdapter()
    for path,dirs,files in tqdm(os.walk(input_path)):
        print("Working on: "+str(path))
        ad.dumpWholeDirectory(path)
    print("Finished!")