# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:30:36 2020

@author: ManavChordia
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:37:26 2020

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
        ent = ent + i * math.log(i, 2)
        
    return -ent

thing=[0.2, 0.4, 0.3, 0.1]
opt = entropy(thing)

print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

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

"""
model.fit(x_train, y_train, epochs=20, verbose=1)
y_test = model.predict(x_test)

all_entropies = []
for i in y_test:
    all_entropies.append(entropy(i))
"""
tot_cntr = 0
num_iter = 0


while 1: 
    print("in while")
    
    if num_iter == 0:
        model.fit(x_train, y_train, validation_data=(x_test, y_true), epochs=20, verbose=1)
        y_test = model.predict(x_test)
        num_iter = num_iter + 1
        
        all_entropies = []
        for i in y_test:
            all_entropies.append(entropy(i))
    
    else:
        print("in else 1")
        
        counter = 0
        not_to_be_removed = []
        not_to_be_removed_y = []
        to_be_removed = []
        for i in range(0, len(all_entropies)):
            if all_entropies[i] > 2.5 and counter < 10 and tot_cntr < 0:

                im = Image.fromarray((x_test[i]*255).astype(np.uint8).reshape(28, 28))
                im.show()
                
                y_in = int(input("enter correct class for x_test[" + str(i) + "]"))

                counter = counter + 1
                tot_cntr = tot_cntr + 1
                temp = []
                for j in range(0, 10):
                    if y_in == j:
                        temp.append(1.)
                    else:
                        temp.append(0.)
                
                y_train = np.append(y_train, np.array(temp).reshape(1, -1), axis = 0)   
                x_train = np.concatenate((x_train, x_test[i].reshape(1, 28, 28, 1)))
                to_be_removed.append(i)
                
            else:
                not_to_be_removed.append(x_test[i])
                not_to_be_removed_y.append(y_true[i])
        x_test = np.array(not_to_be_removed) 
        y_true = np.array(not_to_be_removed_y)
                
        if len(to_be_removed) == 0:
            break

        
        model.fit(x_train, y_train, validation_data=(x_test, y_true) , epochs=20, verbose=1)
        y_test = model.predict(x_test)
        num_iter = num_iter + 1
        print("num _t iter = " + str(num_iter))
        
        all_entropies = []
        for i in y_test:
            all_entropies.append(entropy(i))
            
## accuracy - ~91%

#########################################################
"""
number of samples asked to annotate - 1) 23
                                        2) 24
number of iterations = 3,3
25 samples over 6 iterations

for random - 60 samples and 6 iterations, 57 samples over 7 iterations
"""
#########################################################

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
        
###########################
"""
TESTING ON 6000=8000 FROM TEST DATA
training accuracy - ~98%
incorrect - 131, 136
accuracy - 93.45%
"""  
############################
        
