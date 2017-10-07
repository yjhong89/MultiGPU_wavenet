from __future__ import print_function

import numpy as np
import tensorflow as tf
import os, csv
import ops


WAVE_FILE_TYPE = 'wav'
LABEL_FILE_TYPE = 'txt'
FILE_PATH = 'multigpu_sr/files'
PARENT_PATH = '/home/intel/yjhong89'
NUM_LABELS = 29
NUM_FEATURES = 39

def process_input(data_category):
    wave_file_list = list()
    label_file_list = list()
    # delimiter : distinguish between written sentences
    
    file_path = os.path.join(PARENT_PATH, FILE_PATH)
    csv_f = open(os.path.join(file_path, data_category+'.csv'), 'w') 
    writer = csv.writer(csv_f, delimiter=',')
    '''
        inputs.mel_freq contains mfcc values for each wave file
        inputs.target_label = contains indexed transcript for corresponding wave file
    '''
    data_path = os.path.join(PARENT_PATH, data_category) 
    inputs = ops.SpeechLoader(data_path, NUM_FEATURES, NUM_LABELS)
    mfcc_npy_path = os.path.join(file_path, 'mfcc')
    if not os.path.exists(mfcc_npy_path):
        os.mkdir(mfcc_npy_path)
    
    for i, (wave, label) in enumerate(zip(inputs.mel_freq, inputs.target_label)):
        data_index = '%s_%d' % (data_category, i+1)
        # label is a numpy array
        # Change it to list to 'add' with [data_index]
        writer.writerow([data_index] + label.tolist())
        np.save(mfcc_npy_path + '/' + data_index + '.npy', wave)
        

def read_inputs(data_category, shuffle=False):
    label = list()
    mfcc_file = list()
    file_path = os.path.join(PARENT_PATH, FILE_PATH)
    csv_path = os.path.join(file_path, data_category+'.csv')
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            # MFCC FILE
            mfcc_file.append(os.path.join(PARENT_PATH, 'mfcc') + '/' + row[0] + '.npy')
            # Corresponding LABEL
            label.append(np.asarray(row[1:], dtype=np.int).tostring())

    label_tensor = tf.convert_to_tensor(label)
    mfcc_file_tensor = tf.convert_to_tensor(mfcc_file)

    # tf.train.slice_input_producer : Produces a slice a each 'Tensor' in tensor_list
    label_q, mfcc_file_q = tf.train.slice_input_producer([label_tensor, mfcc_file_tensor], shuffle=shuffle)

    return label_q, mfcc_file_q

def _generate_wave_label_batch(wave, label, min_queue_examples, batch_size, shuffle=True, num_threads=10):
    '''
        Construct a queued batch of wave and label
        Create a queue that shuffles the examples, and pull out batch size wave+label from example queue
    '''

    if not shuffle:
        wave, label = tf.train.batch([wave, label], batch_size=batch_size, num_threads=num_threads, \
                                    capacity=min_queue_examples+20*batch_size)
    else:
        wave, label = tf.train.shuffle_batch([wave, label], batch_size=batch_size, num_threads=num_threads, \
                                capacity=min_queue_examples+20*batch_size, min_after_dequeue=min_queue_examples)

    return wave, label

@ops.producer_func
def _load_mfcc(src_list):
    # Label, wave_file
    label, mfcc_file = src_list               
    # Decode string to integer
    label = np.fromstring(label, np.int)
    # Numpy load mfcc
    mfcc = np.load(mfcc_file)

    return label, mfcc
 
 
def get_batches(data_category, batch_size, num_gpu, num_threads=10, shuffle=False):
    min_queue_examples = batch_size*10
    capacity = batch_size*50
    label_input, wave_input = read_inputs(data_category, shuffle=shuffle)
    wave_list, label_list, seq_len_list = [], [], []

    # Make batches for each gpu
    for i in range(num_gpu):
        label_q, wave_q = _load_mfcc(source=[label_input, wave_input], dtype=[tf.int32, tf.float32], capacity=capacity, num_threads=num_threads)
        waves, labels = _generate_wave_label_batch(wave=wave_q, label=label_q, min_queue_examples=min_queue_examples, batch_size=batch_size, shuffle=shuffle, num_threads=num_threads)
        # Padding for input and make label sparse tensor
        padded_wave, wave_seq_len = ops.pad_sequences(waves)
        sparse_label = ops.sparse_tensor_form(labels)
        wave_list.append(padded_wave)
        label_list.append(sparse_label)
        seq_len_list.append(wave_seq_len)

    return wave_list, label_list, seq_len_list
    

if __name__ == "__main__":
   process_input('train') 
