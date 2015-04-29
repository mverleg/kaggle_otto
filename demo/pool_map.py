
from functools import partial
from multiprocessing import Pool
from time import sleep


class Test(object):
	def __init__(self):
		self.li = []
	def go(self, k):
		self.li.append(k)
		print self.li


def procfunc(k, test):
	test.go(k)
	sleep(2)


floep = Test()
pool = Pool(processes = 4)
print pool.map(partial(procfunc, test = floep), range(20))


