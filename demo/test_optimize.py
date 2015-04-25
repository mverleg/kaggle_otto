
from demo.fake_testing_probabilities import get_from_data, get_random_probabilities
from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator
from validation.optimize import GridOptimizer


train_data, true_classes, features = get_training_data()
validator = SampleCrossValidator(train_data, true_classes, rounds = 5, test_frac = 0.1, use_data_frac = 0.2)
optimizer = GridOptimizer(validator = validator, learning_rate = [10, 1, 0.1, 0.01, 0.001], hidden_layer_size = [60, 30, 50, 40, 20], momentum = 0.9)
#optimizer = GridOptimizer(validator = validator, learning_rate = [1, 0.1, 0.01, 0.001], momentum = 0.9)
for parameters, train, classes, test in optimizer.yield_batches():
	prediction = get_from_data(data = test)
	prediction = get_random_probabilities(sample_count = test.shape[0])
	optimizer.register_results(prediction)
optimizer.print_plot_results()


"""
	Every round has different samples (not completely disjoint), but every e.g. 3rd round has the same, independent of
	parameters. Notice how the loss is exactly the same for deterministic models if the parameters have no effect.
"""


