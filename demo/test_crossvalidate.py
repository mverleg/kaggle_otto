
from demo.fake_testing_probabilities import get_from_data
from utils.loading import get_training_data
from validation.crossvalidate import SampleCrossValidator


train_data, true_classes, features = get_training_data()
validator = SampleCrossValidator(train_data, true_classes, test_frac = 0.1, use_data_frac = 1., show = True)
for train, classes, test in validator.yield_cross_validation_sets(rounds = 15):
	prediction = get_from_data(data = test)
	validator.add_prediction(prediction)
validator.print_results()


