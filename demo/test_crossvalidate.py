
from demo.fake_testing_probabilities import get_random_probabilities
from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator


train_data, true_classes, features = get_training_data()
validator = SampleCrossValidator(train_data, true_classes, test_frac = 0.3, use_data_frac = 0.7)
for train, classes, test in validator.yield_cross_validation_sets(rounds = 13):
	prediction = get_random_probabilities(sample_count = test.shape[0])
	validator.add_prediction(prediction)
validator.print_results()


