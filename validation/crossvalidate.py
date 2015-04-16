
"""
	These functions let you do cross validation as simple as possible (or as simple as I could come up with). Use like this:

		from validation.crossvalidate import get_crossvalidation_data_pairs, calculate_results_for_pairs
		cross_data = get_crossvalidation_data_pairs(N = 10)
		cross_predictions = []
		for train, test in cross_data:
			model = your_training_code(train)
			prediction = your_testing_code(test)
			cross_predictions.append(prediction)
		results = calculate_results_for_pairs(cross_predictions, cross_data)

	See discussion on issue https://gitlab.science.ru.nl/maverickZ/kaggle-otto/issues/6
"""


def get_crossvalidation_data_pairs(rounds = 3, test_frac = 0.3, data_use = 1., seed = 4242):
	pass


def calculate_results_for_pairs():
	pass


def show_results_for_pairs():
	pass

