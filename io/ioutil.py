# -*- coding: utf-8 -*-

import numpy as np

#Reads the test data (the data without class label), and converts it to a numpy matrix
#Removes the first column with the id's.
def readTest(testpath):
    test = np.genfromtxt(testpath, dtype = 'int16', delimiter = ',', skip_header = 1)
    return test[:,1:] #remove the id column

#Reads the train data (With class label), and converts it to numpy feature matrix, and a numpy vector containing the class labels
def readTrain(trainpath):
    train = np.genfromtxt(trainpath, dtype = '|S7', delimiter = ',', skip_header = 1)
    labels = train[:,-1]
    labels = [int(l[-1]) for l in labels] #This assumes the labels are one digit (which they are, 1-9)
    train = train[:,1:-1].astype('int16')
    return train, labels

#Utility, reads all the data given train and test paths
def readData(testpath, trainpath):
    test = readTest(testpath)
    train, labels = readTrain(trainpath)
    return test, train, labels

submissionheader = 'id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9'
    
#Make a submission, given a matrix probs with the class probabilities
#probs is a Nx9 matrix with the probabilities for each class
#fname is where the result is saved
#sampleNrs is what numbers the samples in probabilities have. By default, it just starts counting up from 1
#digits is the number of digits to include after the comma in the submission file.
def makeSubmission(probs, fname = 'results.csv', sampleNrs = None, digits = 5):
    numsamples = np.shape(probs)[0]
    if sampleNrs is None:
        sampleNrs = np.arange(1, numsamples + 1)
    write = np.hstack((np.reshape(sampleNrs, (numsamples, 1)), probs))
    fmtstring = '%0.' + str(digits) + 'f'
    from itertools import repeat
    fmt=[x for x in repeat(fmtstring, 9)]
    fmt.insert(0, '%0.0f')
    np.savetxt(fname, write, fmt = fmt, delimiter = ',', comments='', header = submissionheader)