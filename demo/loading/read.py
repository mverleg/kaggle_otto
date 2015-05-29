from numpy import asarray


def read_data(fname):
    """
    read_data parses the data in file with name fname. When labels is True,
    it reads the label from the last column and returns it as 3rd element
    in the list of tuples
    """
    infile = open(fname)
    header = infile.readline()
    data = []
    labels = []
    for line in infile.readlines():
        dat = line[:-1].split(',')
        tup = (dat[0], asarray(dat[1:94], dtype=int))
        if len(dat) > 94:
            tup += (dat[94][6],)
        data.append(tup)
    infile.close()
    return data

def read_train(fname='train.csv'):
    """
    returns a list of tuples (id, features, label):
        (str, ndarray of type int, str)
    """
    return read_data(fname)

def read_test(fname='test.csv'):
    """
    returns a list of tuples (id, features):
        (str, ndarray of type int)
    """
    return read_data(fname)
