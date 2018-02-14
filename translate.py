import tensorflow as tf 
import os
import string
import re
import numpy as np
import sys
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tqdm import tqdm
import time
import os
import itertools
import gensim

tf.flags.DEFINE_string('load', None,
                       'Restore a model from the given path.')
tf.flags.DEFINE_string('custom_input', '',
                       'Evaluate the model on the given string.')
tf.flags.DEFINE_bool('skip_train', '',
                       'Skip training.')
tf.flags.DEFINE_bool('gpu', False,
                       'Use GPU.')
tf.flags.DEFINE_integer('max_input_len', 100,
                       'The maximum length of an input sentence (in words).')
FLAGS = tf.flags.FLAGS
#FLAGS(sys.argv)

OUT_DIR = os.path.abspath(os.path.join(os.path.curdir, 'output'))
if FLAGS.load is not None:
    # Use logfile and checkpoint from given path
    RUN_DIR = FLAGS.load
    LOG_FILE_PATH = os.path.abspath(os.path.join(RUN_DIR, 'log.log'))
    CHECKPOINT_FILE_PATH = os.path.abspath(os.path.join(RUN_DIR, 'ckpt.ckpt'))
else:
    RUN_ID = time.strftime('run%Y%m%d-%H%M%S')
    RUN_DIR = os.path.abspath(os.path.join(OUT_DIR, RUN_ID))
    # LOG_FILE_PATH = os.path.abspath(os.path.join(RUN_DIR, 'log.log'))
    CHECKPOINT_FILE_PATH = os.path.abspath(os.path.join(RUN_DIR, 'ckpt.ckpt'))
    os.mkdir(RUN_DIR)
SUMMARY_DIR = os.path.join(RUN_DIR, 'summaries')

#Architecture adapted from https://github.com/danielegrattarola/twitter-sentiment-cnn

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
            sentences = re.split("[.?!\r]|<br */>", doc)
            for sentence in sentences:
                words = sentence.translate(None, string.punctuation).split(" ")
                    
                words = filter(None, words)
                if len(words) > FLAGS.max_input_len:
                    args = [iter(words)] * FLAGS.max_input_len
                    words = list(itertools.izip_longest(*args, fillvalue="<PAD>"))
                else:
                    data.append(words)
    return data

print "loading vocabulary\r"
vocab = open(vocab_path).read().splitlines()


# Add various important numbers and tokens
vocab+= ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "<NUM>", "<UNK>", "<PAD>"]
vocab_inv = {} # Converts words to indexes
for i in xrange(len(vocab)):
    vocab_inv[vocab[i]] = i


print "loading word vector model"

wordvec_model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)  


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

sequence_length = max(len(x) for x in train_pos + train_neg + test_pos + test_neg)

# print train_pos[np.argmax([len(sentence) for sentence in train_pos])]
# print train_neg[np.argmax([len(sentence) for sentence in train_neg])]
# print test_pos[np.argmax([len(sentence) for sentence in test_pos])]
# print test_neg[np.argmax([len(sentence) for sentence in test_neg])]



print "Max length: %u"%sequence_length

print "generating input data"

def index(word):
    if word in vocab_inv:
        return vocab_inv[word]
    if word[0].isdigit():
        return vocab_inv["<NUM>"] #Just a random idea I had that numbers are important to distinguish from words
    return vocab_inv["<UNK>"]

def build_data(x_pos, x_neg):
    x_pos = [np.array([wordvec_model.wv[word] for word in sentence if word in wordvec_model.wv])
                            for sentence in (x_pos)]

    x_neg = [np.array([wordvec_model.wv[word] for word in sentence if word in wordvec_model.wv])
                            for sentence in (x_neg)]


    print x_pos[0].shape

    # Pad sentences
    x_pos = np.array([np.pad(sentence, ((0, sequence_length - sentence.shape[0]), (0, 0)), "mean") for sentence in x_pos if sentence != []], 0)

    x_neg = np.array([np.pad(sentence, ((0, sequence_length - sentence.shape[0]), (0, 0)), "mean") for sentence in x_neg if sentence != []], 0)

    x = np.concatenate(x_pos, x_neg)

    print x.shape

    # for a in x:
    #   print a.shape
    y = np.concatenate(([[0, 1] for _ in xrange(x_pos.shape[0])], [[1, 0] for _ in xrange(x_neg.shape[0])]))
    return (x, y)

x_train, y_train = build_data(train_pos, train_neg)

# print x_train.shape
# print y_train.shape

# print x_train.shape
# print x_train[0]

x_test, y_test = build_data(test_pos, test_neg)

#data_chkpt = open("data.npz", "w")

#np.savez(data_chkpt, x_train=x_train, y_train = y_train, x_test=x_test, y_test = y_test)

batch_size = 128
valid_freq = 1
checkpoint_freq = 1
embedding_size = 300
num_filters = 128
epochs = 50


sequence_length = x_train.shape[1]
num_classes = y_train.shape[1]

vocab_size = len(vocab)
filter_sizes = map(int, '3,4,5'.split(','))
validate_every = len(y_train) / (batch_size * valid_freq)
checkpoint_every = len(y_train) / (batch_size * checkpoint_freq)

device = '/cpu:0'
if FLAGS.gpu:
    device = '/gpu:0'


sess = tf.InteractiveSession()


### DEFINE MODEL


def weight_variable(shape, name):
    """
    Creates a new Tf weight variable with the given shape and name.
    Returns the new variable.
    """
    var = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(var, name=name)


def bias_variable(shape, name):
    """
    Creates a new Tf bias variable with the given shape and name.
    Returns the new variable.
    """
    var = tf.constant(0.1, shape=shape)
    return tf.Variable(var, name=name)


with tf.device(device):
    # Placeholders
    data_in = tf.placeholder(tf.int32, [None, sequence_length, embedding_size], name='data_in')
    data_out = tf.placeholder(tf.float32, [None, num_classes], name='data_out')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    # Stores the accuracy of the model for each batch of the validation testing
    valid_accuracies = tf.placeholder(tf.float32)
    # Stores the loss of the model for each batch of the validation testing
    valid_losses = tf.placeholder(tf.float32)



    #  # Embedding layer
    # with tf.name_scope('embedding'):
    #     W = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
    #                                       -1.0, 1.0),
    #                     name='embedding_matrix')
    #     embedded_chars = tf.nn.embedding_lookup(W, data_in)
    #     embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    # Convolution + ReLU + Pooling layer
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope('conv-maxpool-%s' % filter_size):
            # Convolution Layer
            filter_shape = [filter_size,
                            embedding_size,
                            1,
                            num_filters]
            W = weight_variable(filter_shape, name='W_conv')
            b = bias_variable([num_filters], name='b_conv')
            conv = tf.nn.conv2d(data_in,
                                W,
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name='conv')
            # Activation function
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            # Maxpooling layer
            ksize = [1,
                     sequence_length - filter_size + 1,
                     1,
                     1]
            pooled = tf.nn.max_pool(h,
                                    ksize=ksize,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='pool')
        pooled_outputs.append(pooled)

    # Combine the pooled feature tensors
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Dropout
    with tf.name_scope('dropout'):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    # Output layer
    with tf.name_scope('output'):
        W_out = weight_variable([num_filters_total, num_classes], name='W_out')
        b_out = bias_variable([num_classes], name='b_out')
        network_out = tf.nn.softmax(tf.matmul(h_drop, W_out) + b_out)

    # Loss function
    cross_entropy = -tf.reduce_sum(data_out * tf.log(network_out))

    # Training algorithm
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Testing operations
    correct_prediction = tf.equal(tf.argmax(network_out, 1),
                                  tf.argmax(data_out, 1))
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Validation ops
    valid_mean_accuracy = tf.reduce_mean(valid_accuracies)
    valid_mean_loss = tf.reduce_mean(valid_losses)



# Init session
if FLAGS.load is not None:
    print("loading sentiment model")
    #log('Data processing OK, loading network...')
    saver = tf.train.Saver()
    try:
        saver.restore(sess, CHECKPOINT_FILE_PATH)
    except:
        #log('Couldn\'t restore the session properly, falling back to default ' 'initialization.')
        sess.run(tf.global_variables_initializer())
else:
    print("Initializing network")
    #log('Data processing OK, creating network...')
    sess.run(tf.global_variables_initializer())

# Summaries for loss and accuracy
loss_summary = tf.summary.scalar('Training loss', cross_entropy)
valid_loss_summary = tf.summary.scalar('Validation loss', valid_mean_loss)
valid_accuracy_summary = tf.summary.scalar('Validation accuracy',
                                           valid_mean_accuracy)
summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
tf.summary.merge_all()

########### Training ###############

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    # print data.shape
    # print data[0]
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            #print shuffled_data[start_index:end_index].shape
            yield shuffled_data[start_index:end_index]

if not FLAGS.skip_train:

    print "Generating batches"
    batches = batch_iter(zip(x_train, y_train), batch_size, epochs)
    test_batches = list(batch_iter(zip(x_test, y_test), batch_size, 1))
    my_batch = batches.next()  # To use with human_readable_output()

    # Pretty-printing variables
    global_step = 0
    batches_in_epoch = len(y_train) / batch_size
    batches_in_epoch = batches_in_epoch if batches_in_epoch != 0 else 1
    total_num_step = epochs * batches_in_epoch

    batches_progressbar = tqdm(batches, total=total_num_step,
                               desc='Starting training...')

    for batch in batches_progressbar:
        global_step += 1
        x_batch, y_batch = zip(*batch)

        # Run the training step
        feed_dict = {data_in: x_batch,
                     data_out: y_batch,
                     dropout_keep_prob: 0.5}
        train_result, loss_summary_result = sess.run([train_step, loss_summary],
                                                     feed_dict=feed_dict)

        # Print training accuracy
        feed_dict = {data_in: x_batch,
                     data_out: y_batch,
                     dropout_keep_prob: 1.0}
        accuracy_result = accuracy.eval(feed_dict=feed_dict)
        current_loss = cross_entropy.eval(feed_dict=feed_dict)
        current_epoch = (global_step / batches_in_epoch)

        desc = 'Epoch: {} - loss: {:9.5f} - acc: {:7.5f}'.format(current_epoch,
                                                                 current_loss,
                                                                 accuracy_result)
        batches_progressbar.set_description(desc)

        # Write loss summary
        summary_writer.add_summary(loss_summary_result, global_step)

        # Validation testing
        # Evaluate accuracy as (correctly classified samples) / (all samples)
        # For each batch, evaluate the loss
        if global_step % validate_every == 0:
            accuracies = []
            losses = []
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                feed_dict = {data_in: x_test_batch,
                             data_out: y_test_batch,
                             dropout_keep_prob: 1.0}
                accuracy_result = accuracy.eval(feed_dict=feed_dict)
                current_loss = cross_entropy.eval(feed_dict=feed_dict)
                accuracies.append(accuracy_result)
                losses.append(current_loss)

            # Evaluate the mean accuracy of the model using the test accuracies
            mean_accuracy_result, accuracy_summary_result = sess.run(
                [valid_mean_accuracy, valid_accuracy_summary],
                feed_dict={valid_accuracies: accuracies})
            # Evaluate the mean loss of the model using the test losses
            mean_loss_result, loss_summary_result = sess.run(
                [valid_mean_loss, valid_loss_summary],
                feed_dict={valid_losses: losses})

            valid_msg = 'Step %d of %d (epoch %d), validation accuracy: %g, ' \
                        'validation loss: %g' % \
                        (global_step, total_num_step, current_epoch,
                         mean_accuracy_result, mean_loss_result)
            batches_progressbar.write(valid_msg)
            #log(valid_msg, verbose=False)  # Write only to file

            # Write summaries
            summary_writer.add_summary(accuracy_summary_result, global_step)
            summary_writer.add_summary(loss_summary_result, global_step)

        if checkpoint_every != 0 and global_step % checkpoint_every == 0:
            batches_progressbar.write('Saving checkpoint...')
            #log('Saving checkpoint...', verbose=False)
            saver = tf.train.Saver()
            saver.save(sess, CHECKPOINT_FILE_PATH)

    # Final validation testing
    accuracies = []
    losses = []
    for test_batch in test_batches:
        x_test_batch, y_test_batch = zip(*test_batch)
        feed_dict = {data_in: x_test_batch,
                     data_out: y_test_batch,
                     dropout_keep_prob: 1.0}
        accuracy_result = accuracy.eval(feed_dict=feed_dict)
        current_loss = cross_entropy.eval(feed_dict=feed_dict)
        accuracies.append(accuracy_result)
        losses.append(current_loss)

    mean_accuracy_result, accuracy_summary_result = sess.run(
        [valid_mean_accuracy, valid_accuracy_summary],
        feed_dict={valid_accuracies: accuracies})
    mean_loss_result, loss_summary_result = sess.run(
        [valid_mean_loss, valid_loss_summary], feed_dict={valid_losses: losses})
    #log('End of training, validation accuracy: %g, validation loss: %g' %
    #    (mean_accuracy_result, mean_loss_result))

    # Write summaries
    summary_writer.add_summary(accuracy_summary_result, global_step)
    summary_writer.add_summary(loss_summary_result, global_step)

def evaluate_sentence(sentence):
    """
    Translates a string to its equivalent in the integer vocabulary and feeds it
    to the network.
    Outputs result to stdout.
    """

    sentence = sentence.strip().lower().translate(None, string.punctuation).split(" ")
    if len(sentence) > sequence_length:
        print "sentence too long"
        return
    x_to_eval = np.array([index(word) for word in sentence] + [vocab_inv["<PAD>"]]*(sequence_length - len(sentence)))[None,:]


    # x_to_eval = string_to_int(sentence, vocabulary, max(len(_) for _ in x))
    result = sess.run(tf.argmax(network_out, 1),
                      feed_dict={data_in: x_to_eval,
                                 dropout_keep_prob: 1.0})
    unnorm_result = sess.run(network_out, feed_dict={data_in: x_to_eval,
                                                     dropout_keep_prob: 1.0})
    network_sentiment = 'POS' if result == 1 else 'NEG'
    print network_sentiment, unnorm_result
    #log('Custom input evaluation:', network_sentiment)
    #log('Actual output:', str(unnorm_result[0]))

def get_word_grads(sentence, vocabulary, target):
    """
    Translates a string to its equivalent in the integer vocabulary and feeds it
    to the network.
    Outputs result to stdout.
    """
    x_to_eval = string_to_int(sentence, vocabulary, max(len(_) for _ in x))
    feed_dict = {data_in: x_to_eval,
                 data_out: target,
                 dropout_keep_prob: 1.0}

    grad = tf.gradients(accuracy, embedded_chars) # behold the power of automatic differentiation!



    result = sess.run(tf.argmax(network_out, 1),
                      feed_dict={data_in: x_to_eval,
                                 dropout_keep_prob: 1.0})
    unnorm_result = sess.run(network_out, feed_dict={data_in: feed_dict,
                                                     dropout_keep_prob: 1.0})
    network_sentiment = 'POS' if result == 1 else 'NEG'
    print network_sentiment, unnorm_result
    #log('Custom input evaluation:', network_sentiment)
    #log('Actual output:', str(unnorm_result[0]))

if FLAGS.custom_input != '':
    evaluate_sentence(FLAGS.custom_input)