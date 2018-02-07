import tensorflow as tf 
import os
import string
import re

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

print "loading vocabulary"
vocab = open(vocab_path).readlines()
word_to_int = {};
for i in xrange(len(vocab)):
	word_to_int[vocab[i]] = i

print len(vocab)

print "loading data"

train_pos = load_dir(train_pos_dir)
print "loaded %i positive training examples"%len(train_pos)

train_neg = load_dir(train_neg_dir)
print "loaded %i negative training examples"%len(train_neg)

test_pos = load_dir(test_pos_dir)
print "loaded %i positive test examples"%len(test_pos)

test_neg = load_dir(test_neg_dir)
print "loaded %i negative test examples"%len(test_neg)


