
"""
	This related to the random shuffling performing badly, as discussed here:
		https://mlip-maverickz.slack.com/files/fennovj/F04DRQ6EC/shuffle_fix.py
	and explained here:
		http://stackoverflow.com/a/29684037/723090
"""

from random import shuffle, random
v = sum([[k] * 100 for k in range(10)], [])
print v[:40]
shuffle(v, random = lambda: 0.7)
print v[:40]


def rand_tracker():
	rand_tracker.count += 1
	return random()
rand_tracker.count = 0
shuffle(v, random = rand_tracker)
print 'Random function was called %d times for length %d list.' % (rand_tracker.count, len(v))


