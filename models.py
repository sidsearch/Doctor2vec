import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.callbacks import *
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras import backend as K
import tensorflow as tf
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, auc, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import numpy as np
from attention_utils import get_activations, get_data_recurrent



def vectorize_input(data):
    inputs, queries, output = [], [], []
    for s1, q, answer in data:
        inputs.append([word_idx[w] for w in s1])
        queries.append([word_idx[w] for w in q])
        output.append(word_idx[answer])
    return (pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(output))


def sample_data(n_samples=200):
  vectors = []
  for i in range(len(lstm_output)):
    if y_train[i, 1] ==1:
      vectors.append(lstm_output[i,:])
  index = random.sample(range(len(vectors)), n_samples)
  vectors = np.array(vectors)[index,:]
  return vectors



def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul



def modelV0():
	x=Input(shape=(TIME_STEPS, INPUT_DIM,))
	x1 = Embedding(len(max_code+1), embedding_dim,input_length = MAX_SEQ_LENGTH)(x)
	x2 = Bidirectional(LSTM(embedding_dim, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)(x1)
	attention_mul = attention_3d_block(x2)
    attention_mul = Flatten()(attention_mul)
    output = Dense(128, activation='softmax')(attention_mul)

    # clincal trial embedding
    q_cat = Input(shape=(786,))
   	hidden1 = Dense(128, activation='relu')(q_cat)
   	hidden2 = Dense(256, activation='relu')(hidden1)
    hidden3 = Dense(256, activation='relu')(hidden2)
    hidden1 = Dense(64, activation='relu')(visible)

   	# get text embedding from BERT
   	q_text = Input(shape=(TIME_STEPS, INPUT_DIM,))
   	q_final = keras.layers.Add()([q_cat, q_text])
   	
   	h = Bidirectional(LSTM(EMBEDDING_DIM, dropout = 0.2, return_sequences = True))(q_final)
    l_att = Dense(1,activation='tanh')(h)
    l_att = Flatten()(l_att)
    l_att = Activation('softmax')(l_att)
    l_att = RepeatVector(EMBEDDING_DIM*2)(l_att)
    l_att = Permute([2, 1])(l_att)
    l_att = Multiply()([h, l_att])
    print(l_att.shape)
    l_att = Lambda(lambda x: K.sum(x, axis=1))(l_att)
    preds = Dense(2, activation='softmax')(l_att)
  
    model = Model(sentence_input, preds)
    model.compile(loss=[focal_loss(gamma=2., alpha=.25)], 
                optimizer='adam', metrics=['acc']))
    return model

def modelV1():
	
	x=Input(shape=(TIME_STEPS, INPUT_DIM,))
	x1 = Embedding(len(max_code+1), embedding_dim,input_length = MAX_SEQ_LENGTH)(x)
	x2 = Bidirectional(LSTM(embedding_dim, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)(x1)
	attention_mul = attention_3d_block(x2)
    attention_mul = Flatten()(attention_mul)
    output = Dense(128, activation='softmax')(attention_mul)

    # clincal trial embedding
    q_cat = Input(shape=(786,))
   	hidden1 = Dense(128, activation='relu')(q_cat)
   	hidden2 = Dense(256, activation='relu')(hidden1)
    hidden1 = Dense(64, activation='relu')(visible)

   	# get text embedding from BERT
   	q_text = Input(shape=(TIME_STEPS, INPUT_DIM,))
   	q_final = keras.layers.Add()([q_cat, q_text])
   	
   	h = Bidirectional(LSTM(EMBEDDING_DIM, dropout = 0.2, return_sequences = True))(q_final)
    l_att = Dense(1,activation='tanh')(h)
    l_att = Flatten()(l_att)
    l_att = Activation('softmax')(l_att)
    l_att = RepeatVector(EMBEDDING_DIM*2)(l_att)
    l_att = Permute([2, 1])(l_att)
    l_att = Multiply()([h, l_att])
    print(l_att.shape)
    l_att = Lambda(lambda x: K.sum(x, axis=1))(l_att)
    preds = Dense(2, activation='softmax')(l_att)
  
    model = Model(sentence_input, preds)
    model.compile(loss=[focal_loss(gamma=2., alpha=.25)], 
                optimizer='adam', metrics=['acc']))
    return model

def modelV2():
	
	# patient embedding
	x=Input(shape=(TIME_STEPS, INPUT_DIM,))
	x1 = Embedding(len(max_code+1), embedding_dim,input_length = MAX_SEQ_LENGTH)(x)
	x2 = Bidirectional(LSTM(embedding_dim, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)(x1)
	attention_mul = attention_3d_block(x2)
    attention_mul = Flatten()(attention_mul)
    output = Dense(128, activation='softmax')(attention_mul)

    # clincal trial embedding
    q_cat = Input(shape=(786,))
   	hidden1 = Dense(128, activation='relu')(q_cat)
   	hidden2 = Dense(256, activation='relu')(hidden1)

   	# get text embedding from BERT
   	q_text = Input(shape=(TIME_STEPS, INPUT_DIM,))
   	q_final = keras.layers.Add()([q_cat, q_text])
   	
   	h = Bidirectional(LSTM(EMBEDDING_DIM, dropout = 0.2, return_sequences = True))(q_final)
    l_att = Dense(1,activation='tanh')(h)
    l_att = Flatten()(l_att)
    l_att = Activation('softmax')(l_att)
    l_att = RepeatVector(EMBEDDING_DIM*2)(l_att)
    l_att = Permute([2, 1])(l_att)
    l_att = Multiply()([h, l_att])
    print(l_att.shape)
    l_att = Lambda(lambda x: K.sum(x, axis=1))(l_att)
    preds = Dense(2, activation='softmax')(l_att)
  
    model = Model(sentence_input, preds)
    model.compile(loss=[focal_loss(gamma=2., alpha=.25)], 
                optimizer='adam', metrics=['acc']))
    return model

### compile model- this is just to show the model instantiation
model = modelV0()
model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=1024)


# predict probabilities for test set
yhat_probs = model.predict(X_test, batch_size=1024)
yhat_classes = np.argmax(yhat_probs, axis=1)
yhat_probs = yhat_probs[:,1]
y_classes = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_classes, yhat_classes)
print('Accuracy: %f' % accuracy)
precision = precision_score(y_classes, yhat_classes)
print('Precision: %f' % precision)
recall = recall_score(y_classes, yhat_classes)
print('Recall: %f' % recall)
f1 = f1_score(y_classes, yhat_classes)
print('F1 score: %f' % f1)

precision, recall, thresholds = precision_recall_curve(y_classes, yhat_probs, pos_label=1)
print("PR AUC: ", auc(recall, precision))
matrix = confusion_matrix(y_classes, yhat_classes)
print(matrix)