import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from collections import Counter
from sklearn.utils import shuffle

import time


text_data = pd.read_csv('mrdata.tsv', sep='\t')
text_data.columns =['PhraseId', 'SentenceId', 'Phrase', 'Sentiment']
text_data = shuffle(text_data)
#% matplotlib inline
train_data = np.array(text_data[:])
train_data_1 = train_data[:, [2, 3]]

test_data = np.array(text_data[124848:])
test_data_1 = test_data[:, [2]]

labels = train_data_1[:, [1]]
reviews = train_data_1[:, [0]]
unlabeled_reviews = test_data_1
reviews_processed = []
unlabeled_processed = [] 
for review in reviews:
    review_cool_one = ''.join([char for char in review if char not in punctuation])
    reviews_processed.append(review_cool_one)
    
for review in unlabeled_reviews:
    review_cool_one = ''.join([char for char in review if char not in punctuation])
    unlabeled_processed.append(review_cool_one)
word_reviews = []
word_unlabeled = []
all_words = []
for review in reviews_processed:
    word_reviews.append(review.lower().split())
    for word in review.split():
        all_words.append(word.lower())

for review in unlabeled_processed:
    word_unlabeled.append(review.lower().split())
    for word in review.split():
        all_words.append(word.lower())
    
counter = Counter(all_words)
vocab = sorted(counter, key=counter.get, reverse=True)


vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}

reviews_to_ints = []
for review in word_reviews:
    reviews_to_ints.append([vocab_to_int[word] for word in review])



unlabeled_to_ints = []
for review in word_unlabeled:
    unlabeled_to_ints.append([vocab_to_int[word] for word in review])

reviews_lens = Counter([len(x) for x in reviews_to_ints])

seq_len = 52

features = np.zeros((len(reviews_to_ints), seq_len), dtype=int)
for i, review in enumerate(reviews_to_ints):
    if(len(review) == 0):
        print('1 - Length 0')
    else:
        features[i, -len(review):] = np.array(review)[:seq_len]
    
features_test = np.zeros((len(unlabeled_to_ints), seq_len), dtype=int)
for i, review in enumerate(unlabeled_to_ints):
    if(len(review) == 0):
        print('2 - Length 0')
    else:
        features_test[i, -len(review):] = np.array(review)[:seq_len]
a = 156060

X_train = features[:124848]
y_train = labels[:124848]

X_test = features[124848:]
y_test = labels[124848:]

X_unlabeled = features_test

hidden_layer_size = 128 
number_of_layers = 1 
batch_size = 32
learning_rate = 0.01
number_of_words = len(vocab_to_int) + 1
dropout_rate = 0.8
embed_size = 50 
epochs = 2 

tf.reset_default_graph()

inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
targets = tf.placeholder(tf.int32, [None, None], name='targets')

word_embedings = tf.Variable(tf.random_uniform((number_of_words, embed_size), -1, 1))
embed = tf.nn.embedding_lookup(word_embedings, inputs)

hidden_layer = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)
hidden_layer = tf.contrib.rnn.DropoutWrapper(hidden_layer, dropout_rate)

cell = tf.contrib.rnn.MultiRNNCell([hidden_layer]*number_of_layers)
init_state = cell.zero_state(batch_size, tf.float32)

outputs, states = tf.nn.dynamic_rnn(cell, embed, initial_state=init_state)

prediction = tf.layers.dense(outputs[:, -1], 5)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=prediction))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

currect_pred = tf.equal(tf.argmax(tf.nn.softmax(prediction),1), tf.argmax(targets,1))
accuracy = tf.reduce_mean(tf.cast(currect_pred, tf.float32))

session = tf.Session()

session.run(tf.global_variables_initializer())

def getOneHotEncoding(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

tic = time.clock()
for i in range(epochs):
    c = 0
    training_accurcy = []
    ii = 0
    epoch_loss = []
    while ii + batch_size <= len(X_train) and c<2001:
        c += 1
        X_batch = X_train[ii:ii+batch_size]
        y_batch_1 = y_train[ii:ii+batch_size].reshape(-1, 1)
        y_batch = getOneHotEncoding(y_batch_1.astype(np.int32),5).reshape((batch_size,5))
        a, o, _ = session.run([accuracy, cost, optimizer], feed_dict={inputs:X_batch, targets:y_batch})

        training_accurcy.append(a)
        epoch_loss.append(o)
        ii += batch_size
    print('Epoch: {}/{}'.format(i, epochs), ' | Current loss: {}'.format(np.mean(epoch_loss)),
          ' | Training accuracy: {:.4f}'.format(np.mean(training_accurcy)*100))
toc = time.clock()
timeTaken = toc - tic
print("Time taken: ", timeTaken)

test_accuracy = []
c = 0
ii = 0
while ii + batch_size <= len(X_test):
    c += 1
    X_batch = X_test[ii:ii+batch_size]
    y_batch = y_test[ii:ii+batch_size].reshape(-1, 1)

    a = session.run([accuracy], feed_dict={inputs:X_batch, targets:y_batch})
    
    test_accuracy.append(a)
    ii += batch_size

print("Test accuracy is {:.4f}%".format(np.mean(test_accuracy)*100))

