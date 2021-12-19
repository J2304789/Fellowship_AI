#Use following code to install multilingual Amazon Corpus as ARROW file
#Pip install datasets if using Pip
#conda install datasets if using Anaconda
#from datasets import load_dataset
#dataset=load_dataset('amazon_reviews_multi')

#creating Neural Network using .csv of Amazon Review Corpus
import json
from numpy.core.numerictypes import _construct_lookups 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import csv
import random
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from os import chdir,getcwd,path
import matplotlib.pyplot as plt

def Text_Classification():
    #set values for Tokenizer and Pad_Sequences
    embedding_dim=2
    max_length=100
    trunc_type="post"
    padding_type="post"
    oov_tok="oov"
    training_size=20000

    corpus=[]
    sentences=[]
    labels=[]

    num_sentences=0

    #Opens .csv values and creates a list of Corpus
    with open("Documents/GitHub/Fellowship_AI/amazon_review_full_csv/train.csv",encoding="utf-8") as csvfile:
        reader=csv.reader(csvfile,delimiter=',')
        for row in reader:
            list_item=[]
            
            #appends review
            list_item.append(row[2])

            #appends review score based on row value[0]
            this_label=row[0]
            if this_label=='1':
                list_item.append(1)
            elif this_label=='2':
                list_item.append(2)
            elif this_label=='3':
                list_item.append(3)
            elif this_label=='4':
                list_item.append(4)
            else:
                list_item.append(5)

            num_sentences=num_sentences+1
            corpus.append(list_item)

    #Test code to check if .csv file was read
    #print(num_sentences)
    #print(len(corpus))
    #print(corpus[30])

    #Randomizes the reviews
    random.shuffle(corpus)



    #seperates the reviews and scores into labels and sentences
    for x in range(training_size):
        sentences.append(corpus[x][0])
        labels.append(corpus[x][1])

    #print(sentences)
    print(labels)

    #tokenizes values and processes input data
    tokenizer=Tokenizer(oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)
    word_index=tokenizer.word_index
    vocab_size=len(word_index)

    sequences=tokenizer.texts_to_sequences(sentences)
    padded=pad_sequences(sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)


    #creates training and testing sequences
    training_sequences = padded[0:training_size]
    test_sequences = padded[training_size:]
    training_labels = labels[0:training_size]
    test_labels = labels[training_size:]

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, 5, activation=keras.layers.LeakyReLU(alpha=0.001)),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Conv1D(64, 5, activation=keras.layers.LeakyReLU(alpha=0.001)),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])


    training_padded = np.array(training_sequences)
    training_labels = np.array(training_labels)
    testing_padded = np.array(test_sequences)
    testing_labels = np.array(test_labels)


    history = model.fit(training_padded, training_labels, epochs=100)

    #check structure
    #model.summary()

    #plot results using Matplotlib
    acc=history.history["acc"]
    epoch=range(len(acc))

    plt.plot(epoch,acc,"r",label="training")
    plt.title("Training and Validation Accuracy")
    plt.legend(loc="upper right")
    plt.show()

Text_Classification()