#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:19:16 2019

@author: kun-linho
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from tensorflow.keras import backend
from keras import backend as K
from tensorflow.python.client import device_lib
import tensorflow as tf
import random
import collections
from six import StringIO
from copy import copy
from scipy import stats

## need to incorporate early stop 
## search CV use solver

def extractdata(dataset):
    total_data = dataset.iloc[:, 3:12].apply(lambda x: x.astype(str).str.upper()).values.tolist()
    
    return total_data

def extractbinder(dataset):
    binderData = dataset[dataset['Status']==1].iloc[:,3:12].values.tolist()
    return binderData
    
def encodeWithBLOSUM62(amino_acids):
    ## a function that returns the blosum62 vector given a certain aa 
    return list(BLOSUM62_MATRIX[amino_acids].values)

def blosumEncoding(data):
    ## a function that creates amino acid sequence that encode by blosum62 matrix 
    total_data_row =[]
    for each in data:
    
        eachrow =[]
        for aa in each:
            eachrow = eachrow+ encodeWithBLOSUM62(aa)
    
        total_data_row.append(eachrow)
    
    return pd.DataFrame(total_data_row)

def createRandomPeptide(BinderData,NumberData, withBinder ):
    aa_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    # aa_list_new= np.array(aa_list)
    total_list = []
    if withBinder:
        total_list = BinderData
        total_list =[random.sample(aa_list,9) for i in range(NumberData) if random.sample(aa_list,9) not in total_list]
    else:
        
        total_list =[random.sample(aa_list,9) for i in range(NumberData) if random.sample(aa_list,9) not in total_list]

    return total_list

def writeOuput(outputdf, cutOff, outputName ):
    final_list = outputdf.Sequence[outputdf.Total_rank < cutOff].values.tolist()  
    output= open(outputName,'w')
    for each in final_list:
        output.write(each+'\n')
    
    output.close()

# we need to decide which features might affect the result, 
# so choose the column in advance
# Importing the dataset
COMMON_AMINO_ACIDS = collections.OrderedDict(sorted({
   "A": "Alanine",
   "R": "Arginine",
   "N": "Asparagine",
   "D": "Aspartic Acid",
   "C": "Cysteine",
   "E": "Glutamic Acid",
   "Q": "Glutamine",
   "G": "Glycine",
   "H": "Histidine",
   "I": "Isoleucine",
   "L": "Leucine",
   "K": "Lysine",
   "M": "Methionine",
   "F": "Phenylalanine",
   "P": "Proline",
   "S": "Serine",
   "T": "Threonine",
   "W": "Tryptophan",
   "Y": "Tyrosine",
   "V": "Valine",
   }.items()))
COMMON_AMINO_ACIDS_WITH_UNKNOWN = copy(COMMON_AMINO_ACIDS)
COMMON_AMINO_ACIDS_WITH_UNKNOWN["X"] = "Unknown"
   
AMINO_ACID_INDEX = dict((letter, i) for (i, letter) in enumerate(COMMON_AMINO_ACIDS_WITH_UNKNOWN))

AMINO_ACIDS = list(COMMON_AMINO_ACIDS_WITH_UNKNOWN.keys())
   
BLOSUM62_MATRIX = pd.read_csv(StringIO("""
  A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  X
   A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0  0
   R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3  0
   N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  0
   D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  0
   C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1  0
   Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0
   E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  0
   G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3  0
   H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0
   I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3  0
   L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1  0
   K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0
   M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1  0
   F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1  0
   P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2  0
   S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0
   T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0  0 
   W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3  0
   Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1  0
   V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4  0
   X  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
   """), sep='\s+').loc[AMINO_ACIDS, AMINO_ACIDS].astype("int8")
assert (BLOSUM62_MATRIX == BLOSUM62_MATRIX.T).all().all() 
    
    
dataset = pd.read_csv('G:/MAC_Research_Data/MHC_ANN/Datasource/50101/DLA88-50101_peptides.csv')
                      #DLA-88-50101_peptidesforModel.csv')
total_data = extractdata(dataset)       
binderData = extractbinder(dataset)
y = dataset.iloc[:, 2].values
X = blosumEncoding(total_data).values

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split ## cross validation package is depreciate 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle= True,stratify = y)

import keras
from keras.layers import Dropout

from keras.models import Sequential 
from keras.layers import Dense 
# Dense object will take care to initialize the random number close to 0 ( first ANN step)

classifier = Sequential() # use the sequential layer
## init = kernel_initializer
classifier.add(Dense(units = 80, kernel_initializer = 'uniform', activation = 'relu', input_dim = 189))
classifier.add(Dropout(p = 0.5))
classifier.add(Dense(units = 80, kernel_initializer = 'uniform', activation = 'relu', input_dim = 189))
classifier.add(Dropout(p = 0.5))
classifier.add(Dense(units = 80, kernel_initializer = 'uniform', activation = 'relu', input_dim = 189))
classifier.add(Dropout(p = 0.5))
## here is the output layer
## if we deal with more than 2 categories, the activation function needs to use softmax
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
## batch size means, after # of observation, you want to update your weights
classifier.fit(X_train, y_train, batch_size = 10, epochs = 200)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.9) ## thresheld 0.5


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
