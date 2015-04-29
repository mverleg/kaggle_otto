
from demo.fake_testing_probabilities import get_random_probabilities
from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize_parallel import ParallelGridOptimizer


def train_test(train, classes, test, **parameters):
	prediction = get_random_probabilities(sample_count = test.shape[0])
	return prediction

train_data, true_labels = get_training_data()[:2]
validator = SampleCrossValidator(train_data, true_labels, rounds = 6, test_frac = 0.1, use_data_frac = 1)
optimizer = ParallelGridOptimizer(train_test_func = train_test, validator = validator,  use_cache = True, process_count = 3,
	learning_rate = [10, 1, 0.1, 0.01, 0.001],
	hidden_layer_size = [60, 30, 50, 40, 20],
	weight_decay = 0.1,
	momentum = 0.9
).readygo()


