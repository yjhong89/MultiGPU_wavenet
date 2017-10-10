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
        # dynamic_pad=True for variable length inputs, makes [batch, max time step, features]
        # labels also padded with 0
        wave, label = tf.train.batch([wave, label], batch_size=batch_size, shapes=[(None, 39), (None,)], num_threads=num_threads, \
                                    capacity=min_queue_examples+20*batch_size, dynamic_pad=True)
    else:
        #dtypes = list(map(lambda (x, y): (x.dtype, y.dtype), (wave, label)))
        queue = tf.RandomShuffleQueue(capacity=min_queue_examples+20*batch_size, min_after_dequeue=min_queue_examples, dtypes=[tf.float32, tf.int32])
        enqueue_op = queue.enqueue([wave, label])
        queue_r = tf.train.QueueRunner(queue, [enqueue_op]*num_threads)
        tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, queue_r)
        wave, label = queue.dequeue()
        wave, label = tf.train.batch([wave, label], batch_size=batch_size, shapes=[(None, 39), (None,)], num_threads=num_threads, \
                                    capacity=min_queue_examples+20*batch_size, dynamic_pad=True)
        # tf.train.shuffle_batch does not support dynamic pad option
        #wave, label = tf.train.shuffle_batch([wave, label], batch_size=batch_size, num_threads=num_threads, \
        #                        capacity=min_queue_examples+20*batch_size, min_after_dequeue=miin_queue_examples)

    '''
        tf.reduce_sum(wave, axis=2) makes [batch, max time step], padded part would be 0
        tf.not_equal returns bool of shape [batch, max time step]
        tf.cast casting boolean to integer
        tf.reduce_sum(whole, axis=1) returns of shape [batch, ] and each element stands for sequence length of each batch
    '''
    sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(wave, axis=2), 0.), tf.int32), axis=1)

    return wave, label, sequence_len

@ops.producer_func
def _load_mfcc(src_list):
    # Label, wave_file
    label, mfcc_file = src_list               
    # Decode string to integer
    label = np.fromstring(label, np.int)
    # Numpy load mfcc
    mfcc = np.load(mfcc_file, encoding='bytes')

    return label, mfcc
 
 
def get_batches(data_category, batch_size, num_gpu, num_threads=10, shuffle=False):
    min_queue_examples = batch_size*10
    capacity = batch_size*50
    label_input, wave_input = read_inputs(data_category, shuffle=shuffle)
    wave_list, label_list, seq_len_list = [], [], []
    label_q, wave_q = _load_mfcc(source=[label_input, wave_input], dtype=[tf.int32, tf.float32], capacity=capacity, num_threads=num_threads)

    # Make batches for each gpu
    for i in range(num_gpu):
        waves, labels, seq_len = _generate_wave_label_batch(wave=wave_q, label=label_q, min_queue_examples=min_queue_examples, batch_size=batch_size, shuffle=shuffle, num_threads=num_threads)
        #print(labels.get_shape())
        #print(waves.get_shape())
        indices = tf.where(tf.not_equal(tf.cast(labels, tf.float32), 0.))
        #sparse_label = ops.sparse_tensor_form(labels)
        wave_list.append(waves)
        label_list.append(tf.SparseTensor(indices=indices, values=tf.gather_nd(labels, indices), dense_shape=tf.cast(tf.shape(labels), tf.int64)))
        seq_len_list.append(seq_len)

    return wave_list, label_list, seq_len_list
    

if __name__ == "__main__":
   process_input('valid') 
