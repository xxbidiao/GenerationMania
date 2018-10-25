from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pymongo
import sys
from pymongo import MongoClient
import hashlib
import mmap
import base64
from tqdm import tqdm
import soundfile as sf
from ..MongoDBInterface import *
from pydub import AudioSegment
import sys
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

# create path based on this script.
def o(path):
    return os.path.join(os.path.dirname(__file__), path)

class InstrumentLabelingInterface:
    def __init__(self):

        self.FLAGS = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            '--wav', type=str, default='sample.wav', help='Audio file to be identified.')
        self.parser.add_argument(
            '--graph', type=str, default=o('pretrained_graph.pb'), help='Model to use for identification.')
        self.parser.add_argument(
            '--labels', type=str, default=o('train_dir/conv_labels.txt'), help='Path to file containing labels.')
        self.parser.add_argument(
            '--input_name',
            type=str,
            default='wav_data:0',
            help='Name of WAVE data input node in model.')
        self.parser.add_argument(
            '--output_name',
            type=str,
            default='labels_softmax:0',
            help='Name of node outputting a prediction in the model.')
        self.parser.add_argument(
            '--how_many_labels',
            type=int,
            default=3,
            help='Number of results to show.')
        self.classifier = None
        self.FLAGS, self.unparsed = self.parser.parse_known_args()

    def removeAllLabels(self):
        self.mi = MongoDBInterface()
        self.mi.open()
        self.mi.removeAllLabel()

    def set_target(self, filename):
        self.parser.set_defaults(wav=filename)

    def label(self):
        self.FLAGS, self.unparsed = self.parser.parse_known_args()
        tf.app.run(main=self.main, argv=[sys.argv[0]] + self.unparsed)

    def label_all(self):
        self.FLAGS, self.unparsed = self.parser.parse_known_args()

        tf.app.run(main=self.main_all, argv=[sys.argv[0]] + self.unparsed)

    def label_all_func(self):
        return self.main_all(0)

    def label_one(self,data,ext):
        #print("label_one data len:%d"%len(data))
        return self.main_single(data,ext)

    def load_graph(self, filename):
      """Unpersists graph from file as default graph."""
      with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


    def load_labels(self, filename):
      """Read in labels, one label per line."""
      return [line.rstrip() for line in tf.gfile.GFile(filename)]


    def run_graph(self, wav_data, labels, input_layer_name, output_layer_name,
                  num_top_predictions):
      """Runs the audio data through the graph and prints predictions."""
      with tf.Session() as sess:
        # Feed the audio data as input to the graph.
        #   predictions  will contain a two-dimensional array, where one
        #   dimension represents the input image count, and the other has
        #   predictions per class
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
        predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

        # Sort to show labels in order of confidence
        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        for node_id in top_k:
          human_string = labels[node_id]
          score = predictions[node_id]
          print('%s (score = %.5f)' % (human_string, score))

        return 0

    def get_label_from_graph(self, wav_data, labels, input_layer_name, output_layer_name):
        """Runs the audio data through the graph and returns prediction."""
        with tf.Session() as sess:
            # Feed the audio data as input to the graph.
            #   predictions  will contain a two-dimensional array, where one
            #   dimension represents the input image count, and the other has
            #   predictions per class
            softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
            predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

            #print("len:%d"%len(wav_data))
            #print("pred:%s"%predictions)

            # Return label with highest confidence value
            node_id = predictions.argsort()[-1:][::-1][0]
            return labels[node_id]

    def get_labels_from_graph(self, multi_wav_data, labels, input_layer_name, output_layer_name):
        """Runs the audio data through graphs on multiple servers and returns predictions."""
        with tf.Session() as sess:
            # Feed the audio data as input to the graph.
            #   predictions  will contain a two-dimensional array, where one
            #   dimension represents the input image count, and the other has
            #   predictions per class

            multi_predictions = []
            for wav_data in multi_wav_data:
                softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
                predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})
                node_id = predictions.argsort()[-1:][::-1][0]
                multi_predictions.append(labels[node_id])

            return multi_predictions

    def label_wav(self, wav, labels, graph, input_name, output_name, how_many_labels):
        """Loads the model and labels, and runs the inference to print predictions."""
        if not wav or not tf.gfile.Exists(wav):
            tf.logging.fatal('Audio file does not exist %s', wav)

        if not labels or not tf.gfile.Exists(labels):
            tf.logging.fatal('Labels file does not exist %s', labels)

        if not graph or not tf.gfile.Exists(graph):
            tf.logging.fatal('Graph file does not exist %s', graph)

        labels_list = self.load_labels(labels)

        # load graph, which is stored in the default session
        self.load_graph(graph)

        with open(wav, 'rb') as wav_file:
            wav_data = wav_file.read()

        #self.run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)
        return self.get_label_from_graph(wav_data, labels_list, input_name, output_name)

    def label_multiple_wavs_raw(self, wavs_raw_data, labels, graph, input_name, output_name, how_many_labels):
        """Loads the model and labels, and runs the inference to print predictions."""
        if not labels or not tf.gfile.Exists(labels):
            tf.logging.fatal('Labels file does not exist %s', labels)

        if not graph or not tf.gfile.Exists(graph):
            tf.logging.fatal('Graph file does not exist %s', graph)

        labels_list = self.load_labels(labels)

        # load graph, which is stored in the default session
        self.load_graph(graph)

        wavs_data = []
        for wav_raw in wavs_raw_data:
            test_file = open('temp.' + wav_raw['ext'], 'wb')
            test_file.write(wav_raw['data'])
            test_file.close()
            sound = AudioSegment.from_file('temp.' + wav_raw['ext'])
            sound.set_channels(1)
            sound.channels = 1
            sound.set_frame_rate(16000)
            sound.frame_rate = 16000
            sound.export("temp.wav", format="wav")
            with open("temp.wav", 'rb') as wav_file:
              wavs_data.append(wav_file.read())

        #self.run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)
        return self.get_labels_from_graph(wavs_data, labels_list, input_name, output_name)

    def main(self, _):
        """Entry point for script, converts flags to arguments."""
        print(self.label_wav(self.FLAGS.wav, self.FLAGS.labels, self.FLAGS.graph, self.FLAGS.input_name,
                self.FLAGS.output_name, self.FLAGS.how_many_labels))

    def batch_label_from_raw(self, data_list):
        return self.label_multiple_wavs_raw(data_list, self.FLAGS.labels, self.FLAGS.graph, self.FLAGS.input_name,
                self.FLAGS.output_name, self.FLAGS.how_many_labels)

    def main_batch(self, batch_size=1):
        self.mi = MongoDBInterface()
        self.mi.open()
        print("Processing audio samples...")
        total_processed = 0
        current_batch = []
        for result in tqdm(self.mi.getFileWithoutLabel()):
            current_batch.append(result)
            if (len(current_batch) >= batch_size):
                try:
                    current_labels = self.label_multiple_wavs_raw(current_batch, self.FLAGS.labels, self.FLAGS.graph, self.FLAGS.input_name,
                            self.FLAGS.output_name, self.FLAGS.how_many_labels)
                    for current_label in enumerate(current_labels):
                        self.mi.updateLabelForSample(current_batch[current_label[0]]['hash'],current_label[1])
                        print("labeled %s:%s"%(current_label[1], current_batch[current_label[0]]['hash']))
                    tf.reset_default_graph()
                    total_processed += 1
                    current_batch = []
                except:
                    print("Invalid audio sample: ", result['hash'])
                    current_batch = []
        return total_processed

    def main_single(self,data_input,ext):
        #print(self.FLAGS)
        result = {}
        result['data'] = data_input
        result['ext'] = ext
        try:
            test_file = open('temp.' + result['ext'], 'wb')
            #print("len of test_file:%d"%len(result['data']))
            test_file.write(result['data'])
            test_file.close()
            sound = AudioSegment.from_file('temp.' + result['ext'])
            sound.set_channels(1)
            sound.channels = 1
            sound.set_frame_rate(16000)
            sound.frame_rate = 16000
            sound.export(self.FLAGS.wav, format="wav")

            current_label = self.label_wav(self.FLAGS.wav, self.FLAGS.labels, self.FLAGS.graph, self.FLAGS.input_name,
                                           self.FLAGS.output_name, self.FLAGS.how_many_labels)
            #print("run end")
            #self.mi.updateLabelForSample(result['hash'], current_label)
            #print(current_label, result['hash'])

            tf.reset_default_graph()
            return current_label
        except:
            print("Invalid audio sample: ", result['hash'])
            return -1

    def main_all(self, _):
        """Entry point for script, converts flags to arguments."""
        self.mi = MongoDBInterface()
        self.mi.open()
        print("Processing audio samples...")
        total_processed = 0
        for result in tqdm(self.mi.getFileWithoutLabel()):
            try:
                test_file = open('sample.' + result['ext'], 'wb')
                test_file.write(result['data'])
                test_file.close()
                sound = AudioSegment.from_file('sample.' + result['ext'])
                sound.set_channels(1)
                sound.channels = 1
                sound.set_frame_rate(16000)
                sound.frame_rate = 16000
                sound.export("sample.wav", format="wav")
                data, samplerate = sf.read("sample.wav")
                sf.write("sample.wav", data, 16000, subtype='PCM_16')
                current_label = self.label_wav(self.FLAGS.wav, self.FLAGS.labels, self.FLAGS.graph, self.FLAGS.input_name,
                        self.FLAGS.output_name, self.FLAGS.how_many_labels)
                self.mi.updateLabelForSample(result['hash'],current_label)
                print("labeled %s:%s"%(current_label, result['hash']))
                tf.reset_default_graph()
                total_processed += 1
            except:
                print("Invalid audio sample: ", result['hash'])
        return total_processed

labelingInstance = InstrumentLabelingInterface()
def getLabelingInstance():
    return labelingInstance
