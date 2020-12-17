import gzip
import os
import pickle
import sentencepiece as sp
import tensorflow as tf


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_sp_processor(model_path):
    sp_processor = sp.SentencePieceProcessor()
    sp_processor.Load(model_path)
    return sp_processor


def load_dataset(path, encoding):
    with gzip.open(path , "rt", encoding=encoding) as file:
        dataset = file.readlines()
        return dataset


def load_pickled_object(path):
    loaded_object = pickle.load(open(path, "rb"))
    return loaded_object


def load_sentencepiece_model_proto(path):
    with tf.io.gfile.GFile(path, "rb") as file:
        model_proto = file.read()
    return model_proto


def pickle_dump(obj, path):
    with open(path, "wb") as write_file:
        pickle.dump(obj, write_file)