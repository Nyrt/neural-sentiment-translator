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
import tempfile

tf.flags.DEFINE_string('load', None,
                       'Restore a model from the given path.')
tf.flags.DEFINE_string('custom_input', '',
                       'Evaluate the model on the given string.')
tf.flags.DEFINE_float('get_grads', 0,
                       'Evaluate the model on the given string.')
tf.flags.DEFINE_bool('skip_train', '',
                       'Skip training.')
tf.flags.DEFINE_bool('gpu', False,
                       'Use GPU.')
tf.flags.DEFINE_integer('max_input_len', 30,
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
def load_dir(path, num_samples):
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
                    split_words = list(itertools.izip_longest(*args, fillvalue="<PAD>"))
                    for w in split_words:
                        data.append(w)
                        if len(data) >= num_samples:
                            return data
                else:
                    if len(words) > 0:
                        data.append(words)
                    if len(data) >= num_samples:
                        return data

    return data


def build_data(x_pos, x_neg):
    # print [[word for word in sentence if word not in wordvec_model.wv] for sentence in x_pos]

    print "    converting to vectors"

    x_pos = [[wordvec_model.wv[word] for word in sentence if word in wordvec_model.wv]
                            for sentence in (x_pos)]
    x_neg = [[wordvec_model.wv[word] for word in sentence if word in wordvec_model.wv]
                            for sentence in (x_neg)]

    x_pos = [s for s in x_pos if len(s) != 0] # Let's just remove examples with no recognized words
    x_neg = [s for s in x_neg if len(s) != 0]

    num_pos = len(x_pos)
    num_neg = len(x_neg)
    x = x_pos + x_neg

    # Running out of memory!
    del x_pos
    del x_neg

    print "    padding"
    for i in xrange(len(x)):
        x[i] = np.array(x[i])
        if x[i].ndim == 1:
            x[i] = x[i][:,None]
        x[i] = np.pad(x[i], [(0, sequence_length - x[i].shape[0]), (0, 0)], "constant")
    x = np.array(x)

    print "    generating labels"
    y = np.concatenate(([[0, 1] for _ in xrange(num_pos)], [[1, 0] for _ in xrange(num_neg)]))
    return (x, y)

# Construct vocabulary to avoid loading unnecessisary word vectors

print "loading vocabulary\r"
vocab = open(vocab_path).read()
vocab = vocab.split("\n")


# for line in lines:
#     tmp.write(line + " 1\r")


# # Add various important numbers and tokens
# vocab+= ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "<NUM>", "<UNK>", "<PAD>"]
# vocab_inv = {} # Converts words to indexes
# for i in xrange(len(vocab)):
#     vocab_inv[vocab[i]] = i


print "loading word vector model"

wordvec_model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True, limit=100000)  

if not FLAGS.skip_train:

    # print len(vocab)

    print "loading data\r"

    train_pos = load_dir(train_pos_dir, 100000)
    print "loaded %i positive training examples\r"%len(train_pos)

    train_neg = load_dir(train_neg_dir, 100000)
    print "loaded %i negative training examples\r"%len(train_neg)

    test_pos = load_dir(test_pos_dir, 3000)
    print "loaded %i positive test examples\r"%len(test_pos)

    test_neg = load_dir(test_neg_dir, 3000)
    print "loaded %i negative test examples\r"%len(test_neg)

    sequence_length = max(len(x) for x in train_pos + train_neg + test_pos + test_neg)

    print "Seq len", sequence_length


    # print train_pos[np.argmax([len(sentence) for sentence in train_pos])]
    # print train_neg[np.argmax([len(sentence) for sentence in train_neg])]
    # print test_pos[np.argmax([len(sentence) for sentence in test_pos])]
    # print test_neg[np.argmax([len(sentence) for sentence in test_neg])]



    print "Max length: %u"%sequence_length

    print "building training data"

    # def index(word):
    #     if word in vocab_inv:
    #         return vocab_inv[word]
    #     if word[0].isdigit():
    #         return vocab_inv["<NUM>"] #Just a random idea I had that numbers are important to distinguish from words
    #     return vocab_inv["<UNK>"]
        


    x_train, y_train = build_data(train_pos, train_neg)

    # print x_train.shape
    # print y_train.shape

    # print x_train.shape
    # print x_train[0]
    print "building test data"

    x_test, y_test = build_data(test_pos, test_neg)

#data_chkpt = open("data.npz", "w")

#np.savez(data_chkpt, x_train=x_train, y_train = y_train, x_test=x_test, y_test = y_test)

batch_size = 128
valid_freq = 1
checkpoint_freq = 1
embedding_size = 300
num_filters = 32
epochs = 50
learning_rate = 1e-4
reg = 1e-4
training_dropout = 0.5


sequence_length = FLAGS.max_input_len
num_classes = 2

# vocab_size = len(vocab)
filter_sizes = map(int, '3,4,5'.split(','))
if not FLAGS.skip_train:
    validate_every = len(y_train) / (batch_size * valid_freq)
    checkpoint_every = len(y_train) / (batch_size * checkpoint_freq)

device = '/cpu:0'
if FLAGS.gpu:
    device = '/gpu:0'

print "building model"

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
    data_in = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name='data_in')
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
            # W = weight_variable(filter_shape, name='W_conv')
    embedded_chars_expanded = tf.expand_dims(data_in, -1)

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
            conv = tf.nn.conv2d(embedded_chars_expanded,
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
    regularizer = tf.nn.l2_loss(W_out) + tf.nn.l2_loss(W)
    cross_entropy = -tf.reduce_sum(data_out * tf.log(network_out)) 
    loss = cross_entropy + reg*regularizer

    # Training algorithm
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Testing operations
    correct_prediction = tf.equal(tf.argmax(network_out, 1),
                                  tf.argmax(data_out, 1))
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Validation ops
    valid_mean_accuracy = tf.reduce_mean(valid_accuracies)
    valid_mean_loss = tf.reduce_mean(valid_losses)
    
    grad = tf.gradients(cross_entropy, data_in) # behold the power of automatic differentiation!




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
                     dropout_keep_prob: 1.0 - training_dropout}
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



    x_to_eval = np.array([wordvec_model.wv[word] for word in sentence if word in wordvec_model.wv])


    x_to_eval = np.array([np.pad(x_to_eval, [(0, sequence_length - x_to_eval.shape[0]), (0, 0)], "constant")])

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



def distance(source_word, grad, target_word):
    t_0 = (grad.dot(target_word-source_word) + 1e-9) / (grad.dot(grad) + 1e-9) # Closest point on the line
    if t_0 < 0:
        return np.linalg.norm(target_word - (source_word + t_0*grad))# - t_0
    else:
        return 100#np.linalg.norm(source_word - target_word)

word_lr = 100


vocab = [word for word in vocab if word in wordvec_model.wv]
vocab_vecs = np.array([wordvec_model.wv[word] for word in vocab])

def get_word_grads(sentence, target):
    """
    Translates a string to its equivalent in the integer vocabulary and feeds it
    to the network.
    Outputs result to stdout.
    """
    sentence = sentence.strip().lower().translate(None, string.punctuation).split(" ")
    if len(sentence) > sequence_length:
        print "sentence too long"
        return

    loss = 100


    x_to_eval = np.array([wordvec_model.wv[word] for word in sentence if word in wordvec_model.wv])
    x_to_eval = np.array([np.pad(x_to_eval, [(0, sequence_length - x_to_eval.shape[0]), (0, 0)], "constant")])

    indexes = [i for i in xrange(len(sentence)) if sentence[i] in wordvec_model.wv]

    y_target = np.array([[0.5, 0.5]]) 

    if target > 0:
        y_target = np.array([[0, 1]]) 
    elif target <= 0:
        y_target = np.array([[1, 0]]) 



    while loss > 0.1:
        # x_to_eval = string_to_int(sentence, vocabulary, max(len(_) for _ in x))

        result = sess.run(network_out, feed_dict={data_in: x_to_eval, dropout_keep_prob: 1.0})

        new_loss = abs(result[0][1] - target)
        if loss == new_loss:
            print "unable to translate" 
            break
        loss = new_loss
        print loss



        gradients = sess.run(grad, feed_dict={data_in: x_to_eval, data_out: y_target, dropout_keep_prob: 1.0})
        gradients = gradients[0]
        gradients = gradients[0, :, :]


        gradients = gradients * np.linalg.norm(gradients, -1)

        x_to_eval = x_to_eval[0, :, :]

        # Mask out filler
        gradients = gradients * (np.linalg.norm(x_to_eval, 2, -1) != 0)[:, None]

        # print np.linalg.norm(gradients, 2, -1)

        # print x_to_eval.shape
        # print gradients.shape

        ######### Select top words?

        new_words = x_to_eval - gradients * word_lr
        i_x = 0
        for i in xrange(len(sentence)):
            print sentence[i]
            if i in indexes:
                # print wordvec_model.most_similar([new_words[i_x,:]], topn=10)


                distances = np.zeros(len(vocab))
                print np.linalg.norm(gradients[i_x, :])
                for word in xrange(len(vocab)):
                    if vocab[word] != sentence[i]:
                        distances[word] = distance(x_to_eval[i_x,:], gradients[i_x, :], vocab_vecs[word,:])
                

                print [vocab[w] for w in np.argsort(distances)[:10]], np.sort(distances)[:10]

                i_x += 1

        x_to_eval = new_words[None, :, :]

        break



if FLAGS.custom_input != '':
    if FLAGS.get_grads != 0:
        get_word_grads(FLAGS.custom_input, FLAGS.get_grads)
    else:
        evaluate_sentence(FLAGS.custom_input)

