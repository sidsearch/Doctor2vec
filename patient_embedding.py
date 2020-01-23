from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle


my_regularizer = None
my_epochs = 50


encoding_dim = 100 
input_img = Input(shape=(1000, ))
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=my_regularizer)(input_img)
decoded = Dense(1000, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

x_train = np.load('sample_pat_raw_data.npy') 
x_test = np.load('sample_pat_raw_data_test.npy') 

autoencoder.fit(x_train, x_train, epochs=my_epochs, batch_size=256, shuffle=True, validation_data=(x_test, x_test),
                verbose=2)

encoded_data = encoder.predict(x_test)
decoded_data = decoder.predict(encoded_data)

# save features for patients
np.save('patient_features', encoded_data)

