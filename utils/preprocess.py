
"""
    Functions for preprocessing the data, before classification
"""

from numpy import bincount, array, shape, empty, concatenate
from settings import NCLASSES

def obtain_class_weights(true_classes, weight_calc = "inverted"):
    """ 
    Given the true classes of some samples, gives a vector of sample weights.
    The sample weight is a function of the prior probabity of the class the sample belongs to
    By default, the weight is set to inverted, which means the weight is 1 over the prior probability
    This ensures that every class has the same total weight. For now, this is also the only possibility, 
    but more  can be added
    :param true_classes: the true classes of some samples
    """
    
    def inverted(priorprob):
        return 1.0 / priorprob
    
    funcDict = {"inverted" : inverted}

    weightFunc = funcDict[weight_calc]  
    priorprobs = bincount(true_classes, minlength = NCLASSES)
    priorprobs = priorprobs / sum(priorprobs).astype(float)
    return array([weightFunc(priorprobs[true_classes[i]]) for i in range(len(true_classes))])
    
def undersample(featureMatrix, trueClasses):
    """
    Given a number of samples, reduces the featurematrix to contain an equal number of each class
    namely the number of samples of the minimum occuring class
    Any classes with 0 elements will be ignored
    Also, the samples are just picked in order of appeareance, so if you want it to be random, shuffle the data first
    """
    _, C = shape(featureMatrix)
    count = bincount(trueClasses)
    minsamples = min(count[count != 0])
    result = empty((0,C))
    resultclasses = empty((0),dtype = int)
    for i in range(len(count)):
        newsamples = featureMatrix[trueClasses == i, :][:minsamples, :]
        result = concatenate((result, newsamples), axis=0)
        temp = empty((minsamples))
        temp.fill(i)
        resultclasses = concatenate((resultclasses, temp))
    return result

if __name__ == '__main__':
    print undersample(array([[1,2],[2,3],[4,5],[3,2],[3,1],[2,9],[4,6]]), array([1,2,2,1,2,3,4]))
