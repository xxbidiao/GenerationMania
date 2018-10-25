# BMSParserInterface.py
# This file serves as the interface to bms-js package.

import os
import subprocess
import json
from .Config import config
path_to_bmsscript = config.path_to_bmsscript
parser_filename = config.parser_filename
import shlex

# Generates a command line parsing command for the specific file.
# filename: The filename of the file.
def make_parser_call(filename):
    return 'node '+path_to_bmsscript+parser_filename+' '+shlex.quote(str(filename))

class BMSParserInterface:
    def __init__(self):
        pass

    # Parse BMS into json, the internal BMS format.
    @staticmethod
    def getBMSInfo(filename):
        raw_output = subprocess.check_output(make_parser_call(filename),shell=True).decode('utf8')
        json_output = json.loads(raw_output)
        return json_output