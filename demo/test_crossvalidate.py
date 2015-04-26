
from demo.fake_testing_probabilities import get_from_data, get_uniform_probabilities
from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator


train_data, true_classes, features = get_training_data()
validator = SampleCrossValidator(train_data, true_classes, rounds = 30, test_frac = 0.1, use_data_frac = 0.2)
for train, classes, test in validator.yield_cross_validation_sets():
	prediction = get_from_data(data = test)
	#prediction = get_uniform_probabilities(sample_count = test.shape[0])
	validator.add_prediction(prediction)
validator.print_results()


