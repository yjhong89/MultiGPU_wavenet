#!/usr/bin/python
# -*- coding : utf-8 -*-

import numpy as np
import tensorflow as tf
import os, math
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from six.moves import cPickle
from tensorflow.python.framework import ops
import threading
import functools
from tensorflow.python.platform import tf_logging as logging

scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

# Hypothesis probability for decoding
# All probabilities are assumes log prob
class Hypothesis():
    def __init__(self, p_nb, p_b, prefix_len):
     	# probability for not ending blank
        self.p_nb = p_nb
     	# probability for ending blank
        self.p_b = p_b
        self.prefix_len = prefix_len
    
    @staticmethod
    def exp_sum_log(p1, p2):
        ''' Exponential sum of log, 
     	    Return : log(exp(p1)+exp(p2)
     	'''
        exp_sum = math.exp(p1) + math.exp(p2)
        if exp_sum == 0:
            return float('-inf')
        return math.log(exp_sum)
    
    @staticmethod
    def label_to_index(character, num_label):
        if num_label == 30:
            if character == SpeechLoader.SPACE_TOKEN:
                index = 0
            elif character == SpeechLoader.APSTR_TOKEN:
                index = 27
            elif character == SpeechLoader.EOS_TOKEN:
                index = 28
            else:
                index = ord(character) - SpeechLoader.FIRST_INDEX
        elif num_label == 29:
            if character == SpeechLoader.SPACE_TOKEN:
                index = 0
            elif character == SpeechLoader.APSTR_TOKEN:
                index = 27
            else:
                index = ord(character) - SpeechLoader.FIRST_INDEX
        return index

@staticmethod
def index_to_label(idx, num_label):
    index = idx + SpeechLoader.FIRST_INDEX
    if num_label == 30:
    	if index == ord('a') -1:
    		label = SpeechLoader.SPACE_TOKEN
    	elif index == ord('z') + 1:
    		label = SpeechLoader.APSTR_TOKEN
    	elif index ==  ord('a') + 2:
    		label = SpeechLoader.EOS_TOKEN
    	else:
     		label = ord(index)
    
    elif num_label == 29:
    	if index == ord('a') -1:
    		label = SpeechLoader.SPACE_TOKEN
    	elif index == ord('z') + 1:
    		label = SpeechLoader.APSTR_TOKEN
    	else:
     		label = ord(index)
    
    return label

def word_error_rate(first_string, second_string):
	# levenstein distance
	# hsp1116.tistory.com/41
	
	l = first_string.split()
	r = second_string.split()
	
	l_length = len(l)
	r_length = len(r)
	
	# Make distance table
	d = np.zeros(((l_length + 1), (r_length + 1)))
	# Initialize first column and first row
	for i in range(l_length+1):
		d[i][0] = i
	for j in range(r_length+1):
		d[0][j] = j
	
	for i in range(1, l_length + 1):
		for j in range(1, r_length + 1):
			if l[i-1] == r[j-1]:
				d[i][j] = d[i-1][j-1]
			else:
				sub = d[i-1][j-1]
				rem = d[i-1][j]
				add = d[i][j-1]
				d[i][j] = min(sub, rem, add) + 1
	
	return d[l_length][r_length]

def chr_error_rate(first_string, second_string):
	l = list(first_string.replace(' ',''))
	r = list(second_string.replace(' ',''))

	l_length = len(l)
	r_length = len(r)

	d = np.zeros(((l_length + 1), (r_length + 1)))
	for i in range(l_length+1):
		d[i][0] = i
	for j in range(r_length+1):
		d[0][j] = j

	for i in range(1, l_length + 1):
		for j in range(1, r_length + 1):
			if l[i-1] == r[j-1]:
				d[i][j] = d[i-1][j-1]
			else:
				sub = d[i-1][j-1]
				rem = d[i-1][j]
				add = d[i][j-1]
				d[i][j] = min(sub, rem, add) + 1
 
	return d[l_length][r_length]

'''
LayerNormalization
From https://r2rt.com
'''

class LayerNormalizedLSTM(tf.contrib.rnn.RNNCell):
	# state_is_tuple is always true
	def __init__(self, state_size, forget_bias=1.0, activation=tf.nn.tanh):
		self._state_size = state_size
		self._forget_bias = forget_bias
		self._activation = activation

	@property
	def state_size(self):
		return tf.nn.rnn_cell.LSTMStateTuple(self._state_size, self._state_size)
  
	@property
	def output_size(self):
		return self._state_size

	# When class instance called as if it were functions
	def __call__(self, inputs, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__): # LayerNormalizedLSTM
			c, hidden = state
   
		# Change bias to False since LN will add bias via shift term
		# Linear operation to get weighted sum for each gate
		# Modeling LSTM with peepholes
		concat = tf.nn.rnn_cell._linear([inputs, hidden], 4*self._state_size, False)
  
		i, c_tilda, f, o = tf.split(1, 4, concat)

		# Add layer normalization for each gate
		i = ln(i, scope='i/')
		c_tilda = ln(c_tilda, scope='c_tilda/')
		f = ln(f, scope='f/')
		o = ln(o, scope='o/')
   
		# Calculate new c
		new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * self._activation(c_tilda))

		# Add layer normalization in calculation of new hidden state
		new_hidden = self._activation(ln(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
		new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_hidden)

		# Pass to  new_state as state argument
		return new_hidden, new_state

def ln(tensor, scope=None, epsilon=1e-5):
	'''
	 Computing LN given an input tensor. We get in an input of shape [batch_size * state_size], 
	 LN compute the mean and variance for each individual training example across all it`s hidden dimenstino
	 This gives us mean and var of shape [batch_size * 1]
	'''
	# Layer normalization a 2D tensor along its second dimension(along row, for each training example)
	# keep_dims argument need to be set as True to substract, has a shape of [batch_size, 1] not [batch_size,]
	m, v = tf.nn.moments(tensor, [1], keep_dims=True)
	if not isinstance(scope, str):
		scope = ''
	with tf.variable_scope(scope+'layernorm'):
	# Has size of state_size to pointwise multiplication
		scale = tf.get_variable('scale', shape=[tensor.get_shape()[1]], initializer=tf.constant_initializer(1))
		shift = tf.get_variable('shift', shape=[tensor.get_shape()[1]], initializer=tf.constant_initializer(0))

		LN_INITIAL = (tensor - m)/tf.sqrt(v+epsilon)
	return LN_INITIAL*scale + shift

class SpeechLoader():
    # Define class constant
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    APSTR_TOKEN = '<apstr>'
    APSTR_INDEX = 27 
    EOS_TOKEN = '<eos>'
    EOS_INDEX = 28 
    PUNC_TOKEN = '<punc>'
    BLANK_TOKEN = '<blank>'
    # ord('a') = 97
    FIRST_INDEX = ord('a') - 1
    def __init__(self, data_path, num_features, num_classes):
        print('Data from : %s' % data_path)
        self.num_classes = num_classes
        self.num_features = num_features
        self.filelist = list()
        self.wavfilelist = list()
        self.txtfilelist = list()
        self.eachlabellist = list()
        self.mel_freq = list()
        self.target_label = list()
        self.length_check = list()
        print('Number of classes %d, First INDEX : %d' % (self.num_classes, SpeechLoader.FIRST_INDEX))
        
        
        ''' Returning each file`s absolute path
        path : directory path from root
        dir : directory list below root, if there is no directory returns empty list
        file  : file list below root, if there is no file returns empty list
        '''
        for path, dirs, files in os.walk(data_path):
        	fullpath = os.path.join(os.path.abspath(data_path), path)
        	for f in files:
        		filepath = os.path.join(fullpath, f)
        		self.filelist.append(filepath)
        
        for x in self.filelist:
        	if x[-3:] == 'wav':	
        		self.wavfilelist.append(x)
        	elif x[-3:] == 'txt':
        		self.txtfilelist.append(x)
        	else:
        		print(x)
        		raise Exception('Not wanted file type')
        self.num_files = len(self.wavfilelist)
        
        self.wavfilelist.sort()
        self.txtfilelist.sort()
        print('Number of wav files : %d, Number of txt files : %d' % (len(self.wavfilelist), len(self.txtfilelist)))
        
        # Check sequence each wav,label file
        try:
        	for a, b in zip(self.wavfilelist, self.txtfilelist):
        		wavname = a.split('-')[0]+a.split('-')[-1][:-4]
        		txtname = b.split('-')[0]+b.split('-')[-1][:-4]
        	if wavname == txtname:
        		pass
                                #print(wavname)
        except:
        	raise Exception('Files do not match')
        
        # Pasing each line to represent each label
        for each_text in self.txtfilelist:
        	f = open(each_text, 'r')
        	while True:
        		each_line = f.readline()
        		if not each_line:
        			break
        		self.eachlabellist.append(' '.join(each_line.split()))
        	f.close()
        
        self.get_num_examples(self.wavfilelist, self.eachlabellist, self.num_files, self.num_features)
        self.mel_freq = np.asarray(self.mel_freq)
        self.target_label = np.asarray(self.target_label)
        if len(self.length_check) != 0:
        	print(self.length_check)
        	print('Data prprocess not done for %d' % len(self.length_check))
        	raise Exception('input is longer than output')
        print('Data preprocess done')
    
    def get_num_examples(self, wavlists, labellists, num_examples, num_features):
        for n,(w, l) in enumerate(zip(wavlists, labellists)):
            fs, au = wav.read(w)
            # Extract Spectrum of audio inputs
            melf = mfcc(au, samplerate = fs, numcep = self.num_features, winlen=0.025, winstep=0.01, nfilt=self.num_features)
            #melf = (melf - np.mean(melf))/np.std(melf)
            self.mel_freq.append(melf)
            melf_target = self.labelprocessing(l)
            self.target_label.append(melf_target)
            if n == num_examples - 1:
            	break
            if melf.shape[0] <= len(melf_target):
            	t = w,l
            	self.length_check.append(t) 
      
     # Split transcript into each label
    def labelprocessing(self, labellist):
        # Label preprocessing
        label_prep = labellist.lower().replace('[', '')
        label_prep = label_prep.replace(']', '')
        label_prep = label_prep.replace('{', '')
        label_prep = label_prep.replace('}', '')
        label_prep = label_prep.replace('-', '')
        label_prep = label_prep.replace('noise', '')
        label_prep = label_prep.replace('vocalized','')
        label_prep = label_prep.replace('_1', '')
        label_prep = label_prep.replace('  ', ' ')
        trans = label_prep.replace(' ','  ').split(' ')
        labelset = [SpeechLoader.SPACE_TOKEN if x == '' else list(x) for x in trans]
        if self.num_classes == 30:
        	labelset.append(SpeechLoader.EOS_TOKEN)
        	# Make array of each label
        for sentence in labelset:
        	for each_idx, each_chr in enumerate(sentence):
        		if each_chr == "'":
        			sentence[each_idx] = SpeechLoader.APSTR_TOKEN
        labelarray = np.hstack(labelset)
        # Transform char into index
        # including space, apostrophe, eos
        if self.num_classes == 30:
        	train_target = np.asarray([SpeechLoader.SPACE_INDEX if x == SpeechLoader.SPACE_TOKEN else SpeechLoader.APSTR_INDEX \
        		if x == SpeechLoader.APSTR_TOKEN else SpeechLoader.EOS_INDEX if x == SpeechLoader.EOS_TOKEN \
        		else ord(x) - SpeechLoader.FIRST_INDEX for x in labelarray])
        #Including space, apostrophe
        elif self.num_classes == 29:
        	train_target = np.asarray([SpeechLoader.SPACE_INDEX if x == SpeechLoader.SPACE_TOKEN else SpeechLoader.APSTR_INDEX if x == SpeechLoader.APSTR_TOKEN else ord(x) - SpeechLoader.FIRST_INDEX for x in labelarray])
        
        return train_target
    


def sparse_tensor_form(sequences):
    ''' Creates sparse tensor form of sequences
    Argument sequences : A list of lists where each element is a sequence
    Returns : 
      A tuple of indices, values, shape
    '''
    indices = []
    values = []
    
    # Parsing elements in sequences as index and value 
    for n, element in enumerate(sequences):
    	indices.extend(zip([n]*len(element), range(len(element))))
    	values.extend(element)
    
    indices = np.asarray(indices)
    values = np.asarray(values)
    # Python max intenal function : max(0) returns each column max, max(1) returns each row max
    # Need '[]' because it is array, if there is not, it does not a shape ()
    shape = np.asarray([len(sequences), indices.max(0)[1]+1])
    
    return indices, values, shape

def reverse_sparse_tensor(sparse_t):
    '''
    	Input : sparse tensor (indices, value, shape)
    '''
    sequences = list()
    indices = sparse_t[0]
    value = sparse_t[1]
    shape = sparse_t[2]
    	
    start = 0
    # shape[0] : number of sequence
    for i in range(shape[0]):	
        # Get i-th sequence index
        seq_length = len(list(filter(lambda x: x[0] == i, indices)))
        # Use append method instead of extend method
        # Since extend method returns each element iteratively so each element is not seperated
        sequences.append(np.asarray(value[start:(start+seq_length)]))
        start += seq_length
    
    return sequences


def pad_sequences(sequences, max_len=None, padding='post', truncated='post', values=0):
    ''' Pad each sequences to have same length to max_len
    	If max_len is provided, any sequences longer than max_len is truncated to max_len
     	Argument seqeunces will be a array of sequence which has different timestep, last_dimension is num_features
    Returns : 
    	padded sequence, original of each element in sequences 
    '''  
    num_element = len(sequences)
    each_timestep = np.asarray([len(x) for x in sequences])
    
    # Define max_len
    if max_len is None:
    	max_len = np.max(each_timestep)
    
    # Need to add feature size as another dimension
    feature_size = tuple()
    feature_size = np.asarray(sequences[0]).shape[1:]
    
    # Make empty array to bag padded sequence
    x = (np.ones((num_element, max_len) + feature_size) * values).astype(np.float32)
    
    for i, j in enumerate(sequences):
        if len(j) == 0:
            continue
        # Cut post side
        if truncated == 'post':
        	trunc = j[:max_len]
        # Cut pre side
        elif truncated == 'pre':
        	trunc = j[-max_len:]
        else:
        	raise ValueError('Truncated type not supported : %s' % truncated)
        
        # Check shape
        trunc = np.asarray(trunc, dtype=np.float32)
        if trunc.shape[1:] != feature_size:
        	raise ValueError('Shape of truncated sequence %s and expected shape %s is not match' % (trunc.shape[1:], feature_size))
        
        # Substitute original value to 'x'
        if padding == 'post':
        	x[i,:len(j)] = trunc
        elif padding == 'pre':
        	x[i,-len(j):] = trunc
        else:
        	raise ValueError('Padding type not supported : %s' % padding)
        
    return x, each_timestep


def conv1d(inputs, out_channels, filter_width = 2, stride = 1, name = None, activation = tf.nn.relu, normalization = None):
    with tf.variable_scope(name):
        # inputs : [batch, width, channel](1-D)
        w = tf.get_variable('weight', shape=(filter_width, inputs.get_shape().as_list()[-1], out_channels), initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('bias', shape=(out_channels,), initializer=tf.constant_initializer(0))
        # If data format 'NHWC' : [batch, in_width, in_channel]
        # If data format 'NCHW' : [batch, in_channel, in_width]
        outputs = tf.nn.conv1d(inputs, w, stride=stride, padding='SAME',data_format='NHWC')
        weighted_sum = outputs + b
        if normalization:
            outputs = tf.contrib.layers.layer_norm(weighted_sum, scope = 'ln')
        if activation:
            outputs = activation(outputs)
        return outputs

def dilated_conv1d(inputs, out_channels, filter_width = 2, padding='SAME', rate = 1, causal = True, name = None, activation = None, normalization = None):
    with tf.variable_scope(name):
        w = tf.get_variable(name='weight', shape=(filter_width, inputs.get_shape().as_list()[-1], out_channels), initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='bias', shape=(out_channels,), initializer=tf.constant_initializer(0.0))
        # causal means output at time 't' only depends on its past and current 't', so pad only left side
        # When causal, need to insert (filter_width - 1) zeros to left side of input
        if causal:
            # Padding 'Same' = with zero padding so make same with filter and input calculation dimension, so pad zeros with inputs
            # Produce output of the same size as the input when stride = 1
            # Padding 'Valid' = without padding, drop right most columns
            # Produce output as 'filter size - 1'
            pad_len = (filter_width-1) * rate
            '''
                tf.pad : 
                    For each dimension D of input, paddings[D,0] indicates how many values to add 'before' the content of tensor in that dimension 'D'
                    padding[D,1] indicates how many values to add 'after' the content of tensor in that dimension 'D'
            '''
            # In dimension '0'(batch axis) no padding
            # In dimension '1'(input width) add value only before the content of tensor becuase it is causal
            # In dimension '2'(in channel) no padding
            inputs = tf.pad(inputs, [[0,0],[pad_len, 0], [0,0]])
            '''
                tf.nn.convolution : dilation_rate(Sequence of integers and each index of sequence represents filter axis)
                    The effective filter size will be [filter_shape[i] + (filter_shape[i] - 1) * (rate[i] - 1)]
                    obtained by inserting (rate[i] - 1] zeros between consecutive elements of the original filter
                    Stridse must be 1 when rate is larger than 1
            '''
            # Here, dilation_rate = [rate], which represents insert zeros only filter`s 0 axis
            outputs = tf.nn.convolution(inputs, w, dilation_rate = [rate], padding = 'VALID') + b
        # Since non causal, pad zeors also to right part of input
        else:
            outputs = tf.nn.convolution(inputs, w, dilation_rate = [rate], padding = 'SAME') + b
        if normalization:
            outputs = tf.contrib.layers.layer_norm(outputs, scope = 'ln')
        if activation:
            outputs = activation(outputs)
    return outputs

# Residual block implementation of wavenet(Figure 4 in 2.3/2.4)
# Different parameters with different layers
# Works better than relu activation for modeling audio signal
def res_block(tensor, num_hidden, rate, causal, dilated_filter_width, normalization, activation, name=None):
	# Gated Actiation Unit
	if activation == 'gated_linear':
		act = None
	elif activation == 'gated_tanh':
		act = tf.nn.tanh
	with tf.variable_scope(name):
		h_filter = dilated_conv1d(tensor, num_hidden, rate = rate, causal = causal, name = 'conv_filter', activation = act, normalization = normalization)
		h_gate = dilated_conv1d(tensor, num_hidden, rate = rate, causal = causal, name = 'conv_gate', activation = tf.nn.sigmoid, normalization = normalization)
		# Elementwise multiication
		out = h_filter * h_gate    
		# Generate residual part and skip part through 1*1 convolution
		residual = conv1d(out, num_hidden, filter_width=1, activation=None, normalization=normalization, name='res_conv')   # (batch, width, num_hidden)
		skip = conv1d(out, num_hidden, filter_width=1, activation=None, normalization=normalization, name='skip_conv')   # (batch, width, num_hidden)

	# Return skip part and residual for next layer
	return tensor + residual, skip

'''
    From : 'https://github.com/buriburisuri/sugartensor/blob/master/sugartensor/sg_queue.py'
'''

def producer_func(func):
    '''
        func : A function to decorate
        functools.wrap : To maintain name space of original function
        From : devspark.tistory.com/entry/functoolswraps에_대해
    '''

    @functools.wraps(func)
    def wrapper(**kwargs):
        '''
            Args:
                source: A source list queue list to enqueue
                dtypes: Input data types of each tensor
                capacity: Queue capacity
                num_threads: 1
        '''
        
        opt = kwargs
        # Skip source list check
        
        # Enqueue function
        def enqueue_func(sess, op):
            # Read data from source queue
            data = func(sess.run(opt['source']))
            # Create feeder dict
            feed_dict = {}
            for place_h, d in zip(placeholders, data):
                feed_dict[place_h] = d
            # Run session
            sess.run(op, feed_dict=feed_dict)

        # Create placeholder
        placeholders = list()
        for dtype in opt['dtype']:
            placeholders.append(tf.placeholder(dtype=dtype))

        # Create FIFO queue
        queue = tf.FIFOQueue(opt['capacity'], dtypes=opt['dtype'])

        # Enqueue opration
        enqueue_op = queue.enqueue(placeholders)

        # Create queue runner, pass enqueue function to queue runner
        runner = _FuncQueueRunner(enqueue_func, queue, [enqueue_op]*opt['num_threads'])

        # Register to global collection
        tf.train.add_queue_runner(runner)

        # Rerurn dequeue operation
        return queue.dequeue()

    return wrapper

class _FuncQueueRunner(tf.train.QueueRunner):
    def __init__(self, func, queue=None, enqueue_ops=None, close_op=None, cancel_op=None, \
                queue_closed_exception_types=None, queue_runner_def=None):

        self.func = func
        super(_FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op, queue_closed_exception_types, queue_runner_def)

    def _run(self, sess, enqueue_op, coord=None):
        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    # Call enqueue_op
                    self.func(sess, enqueue_op) 
                except self._queue_closed_exception_types:
                    # This exception indicates that a queue was closed
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
                except ValueError:
                    pass
        except Exception as e:
            # This catches all other exceptions
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we acount for all terminations
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1
                    


if __name__ == '__main__':
	'''
		Checking function 
	'''
	a = np.array([1,2,3,4])
	b = np.array([5,6,7])
	c = np.array([8,9])
	d = [a,b,c]
	d = np.asarray(d)
	print(a,b,c,d)
	e = sparse_tensor_form(d)
	print(e[0], e[1], e[2])
	f = reverse_sparse_tensor(e)
	print(f)
	# a,b,c = SpeechLoader.sparse_tensor_form(asr.target_label)
	# print a,b,c
	
	# timestep = np.random.randint(0,10, (3,))
	# i = np.asarray([np.random.randint(0,10,(t, 13)) for t in timestep])
	# x,y = SpeechLoader.pad_sequences(i)
	# print x,y
	# print type(y), y.shape
	
	# f = ''
	# s = 'who is there'
	# print  chr_error_rate(f,s)





  

