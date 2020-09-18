# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:31:31 2020

@author: ManavChordia
"""

############### with class imbalance ################################

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
            i = i + 0.00001
        ent = ent + i * math.log(i, 2)
        
    return -ent


def extract_index(num, how_many, temp):
    l_y = []
    l_x = []
    cnt = 0
    if temp == 0:
        for i in range(0, len(training_labels)):
            if training_labels[i] == num and cnt < how_many:
                l_y.append(training_labels[i])
                l_x.append(training_images[i])
                cnt = cnt+1
    
    else:
        for i in range(0, len(test_labels)):
            if test_labels[i] == num and cnt < how_many:
                l_y.append(test_labels[i])
                l_x.append(test_images[i])
                cnt = cnt+1
            

    return [np.array(l_y), np.array(l_x)]
    

thing_1 = [0.3, 0.4, 0.2, 0.0]
opt_1 = entropy(thing_1)
thing_2 = [0.01, 0.005, 0.005, 0.98]
opt_2 = entropy(thing_2)


######## increase 5 and 3 reduce 8 and increase 1 and 9 and reduce 4 #######################
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b



print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

####create training data
y_8, x_8 = extract_index(8, 40, 0)
y_5, x_5 = extract_index(5, 7, 0)
y_3, x_3 = extract_index(3, 13, 0)
y_1, x_1 = extract_index(1, 30, 0)
y_9, x_9 = extract_index(9, 20, 0)
y_4, x_4 = extract_index(4, 10, 0)
y_2, x_2 = extract_index(2, 20, 0)
y_6, x_6 = extract_index(6, 30, 0)
y_7, x_7 = extract_index(7, 20, 0)
y_0, x_0 = extract_index(0, 10, 0)

x_train = np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), axis = 0)
y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis = 0)
x_train, y_train = shuffle_in_unison(x_train, y_train)

####create testing data
y_8, x_8 = extract_index(8, 400, 1)
y_5, x_5 = extract_index(5, 70, 1)
y_3, x_3 = extract_index(3, 130, 1)
y_1, x_1 = extract_index(1, 300, 1)
y_9, x_9 = extract_index(9, 200, 1)
y_4, x_4 = extract_index(4, 100, 1)
y_2, x_2 = extract_index(2, 200, 1)
y_6, x_6 = extract_index(6, 300, 1)
y_7, x_7 = extract_index(7, 200, 1)
y_0, x_0 = extract_index(0, 100, 1)

x_test = np.concatenate((x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9), axis = 0)
y_true = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis = 0)
x_test, y_true = shuffle_in_unison(x_test, y_true)


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
            if all_entropies[i] > 2.5 and counter < 10 and tot_cntr < 25:

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

        
        model.fit(x_train, y_train, validation_data=(x_test, y_true), epochs=20, verbose=1)
        y_test = model.predict(x_test)
        num_iter = num_iter + 1
        print("num _t iter = " + str(num_iter))
        
        all_entropies = []
        for i in y_test:
            all_entropies.append(entropy(i))
            
            
#########################################################

#with class imbalance - ~ 53 and 8 iterations, 46 and 8 iterations

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
#TESTING ON 6000=8000 FROM TEST DATA
#training accuracy - ~98%
#incorrect - 124, 141
#accuracy - ~93-94
############################
        
## when restricted to 25 annotable samples - accuracy - ~92%
        