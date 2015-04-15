# -*- coding: utf-8 -*-

import numpy as np


global TRAINSIZE, TESTSIZE, NFEATS

TRAINSIZE=61878
TESTSIZE=144368
NFEATS=93


def _read_data(test='../test', train='../train'):
    """
    read_data('train.csv', 'test.csv')
    
    Parse the training and test data at the given paths.
    
    Parameters
    ----------
    test :  str
            the file containing the test data
            Defaults to '../test' (hard link to data/test.csv)
    train : str
            the file containing the training data.
            Defaults to '../train' (hard link to data/train.csv)
    
    Returns
    -------
    A 3-tuple (ndarray, ndarray, tuple) with the test and training data
    with its labels.
    """
    testfile, trainfile = open(test), open(train)
    def feats(dat):
        return np.asarray(dat[:NFEATS], dtype=np.int16)

    def reader(inf):
        inf.readline()
        for i, line in enumerate(inf.readlines()):
            dat = line[:-1].split(',')[1:]
            yield i, dat
        inf.close()
    
    test = np.zeros((TESTSIZE, NFEATS), dtype=np.int16)
    train = np.zeros((TRAINSIZE, NFEATS), dtype=np.int16)
    labels = np.zeros(TRAINSIZE, dtype=np.int8)

    for i,dat in reader(testfile):
        test[i] = feats(dat)
        
    for i,dat in reader(trainfile):
        train[i] = feats(dat)
        labels[i] = np.int8(dat[NFEATS][6])
    
    return test, train, labels


global TEST, TRAIN, LABELS

TEST, TRAIN, LABELS = _read_data()


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
