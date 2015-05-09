# -*- coding: utf-8 -*-
"""
Created on Sat May 09 22:14:06 2015

@author: Fenno
"""
from random_forest.randomforest import randomForest
from gradient_boosting.gradientboosting import gradientBoosting
from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator
import numpy as np


train, labels, features = get_training_data()

class1 = np.where(labels == 1)[0]
class2 = np.where(labels == 2)[0]
class3 = np.where(labels == 3)[0]
class4 = np.where(labels == 4)[0]
class9 = np.where(labels == 9)[0]

trainindexes = np.concatenate((class2, class3))

newtrain= train[trainindexes,:]
newlabels = labels[trainindexes].astype('int32')

x = np.where(newlabels == np.min(newlabels))[0]
y = np.where(newlabels == np.max(newlabels))[0]
newlabels[x] = 1#np.zeros(len(x), dtype = 'int32')
newlabels[y] = 2#np.ones(len(y), dtype = 'int32')


print len(x)
print len(y)
"""
print newlabels[:10]
print newlabels[-10:]
print y[-10:]
print np.min(newlabels), np.max(newlabels)

print np.shape(newtrain)
print np.shape(newlabels)
"""
validator = SampleCrossValidator(newtrain, newlabels, rounds=1, test_frac=0.1, use_data_frac=1.0)

for train, classes, test in validator.yield_cross_validation_sets():

    prediction = randomForest(train, classes, test, 5, n_estimators = 200 )
    #prediction = gradientBoosting(train, classes, test, n_estimators = 1000)
      
    validator.add_prediction(prediction)
validator.print_results()
