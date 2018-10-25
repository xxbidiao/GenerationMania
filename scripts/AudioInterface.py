import sys
from .sound_classification.InstrumentLabelingInterface import InstrumentLabelingInterface
from .pitch_classification.PitchLabelingInterface import PitchLabelingInterface

# Label all files in the database one by one.
def do_audio(type="instrument"):
    if type == "instrument":
        foo = InstrumentLabelingInterface()
        samples = foo.label_all_func()
        # print("Classified %d samples."%samples)
    elif type == "pitch":
        foo = PitchLabelingInterface()
        samples = foo.label_all_func()

# Label one audio piece.
# data: the audio data.
# ext: the extension (expected format) for the file.
# Note - file format other than WAV/OGG is not fully tested yet.
def do_one_audio(data,ext,type="instrument"):
    if type == "instrument":
        foo = InstrumentLabelingInterface()
        samples = foo.label_one(data,ext)
        return samples
    elif type == "pitch":
        foo = PitchLabelingInterface()
        samples = foo.label_one(data,ext)

# do batch audio classification assuming all files are ogg.
# data_list: a list of audio pieces.
def do_batch_ogg(data_list,type="instrument"):
    input_arr = []
    for data in data_list:
        input_arr.append({'ext':'ogg','data':data})
    return do_batch_audio(input_arr,type)

# Do audio classification for a list of files.
# data_list: list in format [{ext:...,data:...},...]
def do_batch_audio(data_list,type="instrument"):
    if type == "instrument":
        foo = InstrumentLabelingInterface()
        samples = foo.batch_label_from_raw(data_list)
        return samples
