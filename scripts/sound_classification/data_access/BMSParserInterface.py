import os
import subprocess
import json

path_to_bmsscript = '../bmsscript/'
parser_filename = 'main.js'
import shlex

def make_parser_call(filename):
    return 'node '+path_to_bmsscript+parser_filename+' '+shlex.quote(str(filename))

class BMSParserInterface:
    def __init__(self):
        pass

    @staticmethod
    def getBMSInfo(filename):
        raw_output = subprocess.check_output(make_parser_call(filename),shell=True).decode('utf8')
        json_output = json.loads(raw_output)
        return json_output