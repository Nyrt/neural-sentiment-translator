import tensorflow as tf 
import os
import string
import re
import numpy as np

train_pos_dir = "data/aclImdb/train/pos/"
train_neg_dir = "data/aclImdb/train/neg/"

test_pos_dir = "data/aclImdb/test/pos/"
test_neg_dir = "data/aclImdb/test/neg/"

vocab_path = "data/aclImdb/imdb.vocab"


# returns a list of documents, each of which is a list of words
def load_dir(path):
	data = []
	for file in os.listdir(path):
		if file.endswith(".txt"):
			#print os.path.join(path, file)
			doc = open(os.path.join(path, file)).read().strip().lower()#.translate(None, string.punctuation).split(".!?")
			sentences = re.split("\.?!", doc)
			for sentence in sentences:
				words = sentence.translate(None, string.punctuation).split(" ")
				data.append(words)
	return data

print "loading vocabulary\r"
vocab = open(vocab_path).readlines()
vocab_inv = {} # Converts words to indexes
for i in xrange(len(vocab)):
	vocab_inv[vocab[i]] = i

print len(vocab)

print "loading data\r"

train_pos = load_dir(train_pos_dir)
print "loaded %i positive training examples\r"%len(train_pos)

train_neg = load_dir(train_neg_dir)
print "loaded %i negative training examples\r"%len(train_neg)

test_pos = load_dir(test_pos_dir)
print "loaded %i positive test examples\r"%len(test_pos)

test_neg = load_dir(test_neg_dir)
print "loaded %i negative test examples\r"%len(test_neg)

#checkpoint_file = open("model.ckpt", "w")

x_train = np.array([vocab_inv[word] for word in [sentence for sentence in (train_pos + train_neg)]])
y_train = np.hstack(np.ones(len(train_pos)), -np.ones(len(train_neg)))


batch_size = 128
valid_freq = 1
checkpoint_freq = 1

sequence_length = x_train.shape[1]
num_classes = y_train.shape[1]
vocab_size = len(vocabulary)
filter_sizes = map(int, '3,4,5'.split(','))
validate_every = len(y_train) / (batch_size * valid_freq)
checkpoint_every = len(y_train) / (batch_size * checkpoint_freq)

device = '/gpu:0'
sess = tf.InteractiveSession()

with tf.device(device):
    # Placeholders
    data_in = tf.placeholder(tf.int32, [None, sequence_length], name='data_in')
    data_out = tf.placeholder(tf.float32, [None, num_classes], name='data_out')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    # Stores the accuracy of the model for each batch of the validation testing
    valid_accuracies = tf.placeholder(tf.float32)
    # Stores the loss of the model for each batch of the validation testing
    valid_losses = tf.placeholder(tf.float32)