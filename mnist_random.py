# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:41:59 2020

@author: ManavChordia
"""


import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image


def entropy(thing):
    ent = 0
    
    for i in thing:
        if i == 0:
            i = i + 0.0001
        ent = ent + i * math.log(i, 2)
        
    return -ent


            
thing_1 = [0.3, 0.4, 0.2, 0.0]
opt_1 = entropy(thing_1)
thing_2 = [0.01, 0.005, 0.005, 0.98]
opt_2 = entropy(thing_2)

print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


x_train = test_images[0:200]
y_train = test_labels[0:200]


x_train = test_images[0:200]
y_train = test_labels[0:200]

x_test = test_images[4000:5000]
y_true = test_labels[4000:5000]

y_train = y_train.reshape(-1,1)
y_true = y_true.reshape(-1,1)

enc = OneHotEncoder(handle_unknown='ignore')
y_train = enc.fit_transform(y_train).toarray()
y_true = enc.fit_transform(y_true).toarray()


from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, Dropout, Flatten
from keras.models import Model
from keras.models import Sequential
from keras import backend as K

model = Sequential()
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation= 'softmax'))
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) 
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  


#model.fit(x_train, y_train, epochs=20, verbose=1)
#y_test = model.predict(x_test)

#all_entropies = []
#for i in y_test:
#    all_entropies.append(entropy(i))


from random import sample

tot_number = 0
num_iter = 0
while num_iter < 6: 
    print("in while")
    
    if num_iter == 0:
        model.fit(x_train, y_train, epochs=20, verbose=1)
        y_test = model.predict(x_test)
        num_iter = num_iter + 1
        
        all_entropies = []
        for i in y_test:
            all_entropies.append(entropy(i))
    
    else:
        print("in else 1")
        
        counter = 0
        not_to_be_removed = []
        to_be_removed = sample([i for i in range(0, len(x_test))], 5)
        
        for j in to_be_removed:

            x_train = np.concatenate((x_train, x_test[j].reshape(1, 28, 28, 1)))
            y_train = np.concatenate((y_train, y_true[j].reshape(1, -1)))
            
        alt = []
        alt_y = []
        for i in range(0, len(x_test)):
            if i not in to_be_removed:
                alt.append(x_test[i])
                alt_y.append(y_true[i])
                
        x_test = np.array(alt)
        y_true = np.array(alt_y)
        
        
        model.fit(x_train, y_train, epochs=20, verbose=1)
        y_test = model.predict(x_test)
        num_iter = num_iter + 1
        print("num _t iter = " + str(num_iter))
        
        all_entropies = []
        for i in y_test:
            all_entropies.append(entropy(i))   
            
        cnt = 0
        for i in all_entropies:
            if i > 2.5:
                cnt = cnt + 1
        print()
        print(cnt)
        print()
        if cnt == 0:
            break
        
        tot_number = tot_number + 1
        
x_test_new = test_images[6000:8000]
x_test_new = x_test_new.astype('float32') / 255.
x_test_new = np.reshape(x_test_new, (len(x_test_new), 28, 28, 1))  
#model.fit(x_train, y_train, epochs=20, verbose=1)
y_test_new = model.predict(x_test_new)

y_predicted = [np.argmax(y_test_new[i]) for i in range(0,2000) ]
y_true = test_labels[6000:8000]
false = 0
for i in range(0, 2000):
    if y_predicted[i] != y_true[i]:
        false = false + 1
        
        
## when restricted to 25 annotable samples - accuracy - ~90-91%

           
        
        
