import pandas as pd
import numpy as np
import yaml
import os
from typing import Text
import argparse
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import seaborn as sns
import tensorflow as tf

def train(config_path: Text) -> None:
    with open("params.yaml") as config_file:
        config = yaml.safe_load(config_file)
    
    with open('/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/data/movie-xids.npy', 'rb') as f:
        Xids = np.load(f, allow_pickle=True)
    with open('/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/data/movie-xmask.npy', 'rb') as f:
        Xmask = np.load(f, allow_pickle=True)
    with open('/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/data/movie-labels.npy', 'rb') as f:
        labels = np.load(f, allow_pickle=True)

    #Convert three arrays and into TF dataset object using from_tensor_slices
    dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

    print('Tensorflow dataset object showing 1 observation: ', dataset.take(1))

    # Define map function that splits the input tensors from ouput 
    def map_func(input_ids, masks, labels):
        # we convert our three-item tuple into a two-item tuple where the input item is a dictionary
        return {'input_ids': input_ids, 'attention_mask': masks}, labels

    # then we use the dataset map method to apply this transformation
    dataset = dataset.map(map_func)

    print('TF object after splitting input and output tensors: ', dataset.take(1))


#to run from CLI use a constructer that allows to parse config file as an argument to the data_load function
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)