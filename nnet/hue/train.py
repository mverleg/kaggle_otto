
from numpy import float32
from nnet.hue.get_train_data import equalize_train_classes
from nnet.make_net import make_net
from validation.score import calc_logloss, calc_accuracy


def do_train(min_cls_size, trainData, trainLabels, residualData, residualLabels, testData, testLabels, round_nr = 5):
	iterations = [200]
	hidden_numbers = [25]
	learning_rates = [0.1]

	for round in range(round_nr):
		print 'round', round
		for iteration in iterations:
			for hidden_number in hidden_numbers:
				for learning_rate in learning_rates:
					print 'parameter', iteration, 'layer', hidden_number, 'lr', learning_rate
					print 'creating network'
					net = make_net(
						name = 'matlabnet',
						dense1_size = hidden_number,
						dense1_nonlinearity = 'sigmoid',
						dense1_init = 'he_normal',
						dense2_size = None,
						learning_rate_start = learning_rate,
						learning_rate_end = learning_rate,
						momentum_start = 0.9,
						momentum_end = 0.9,
						dropout1_rate = None,
						dropout2_rate = None,
						weight_decay = 0,
						max_epochs = iteration,
						verbosity = True,  # turn output on or off
						auto_stopping = False,  # don't stop automatically
					)

					print 'equalizing classes'
					print trainData[0].dtype
					trainData, trainLabels, residualData, residualLabels = equalize_train_classes(min_cls_size, trainData, trainLabels)

					# convert everything to float32 because float64 does not work
					#trainData, trainLabels, residualData, residualLabels = [arr.astype(float32) for arr in (trainData, trainLabels, residualData, residualLabels)]

					print 'training network'
					print trainData.dtype, trainLabels.dtype
					out = net.fit(trainData, trainLabels - 1)

					print 'predicting test data'
					prediction = net.predict_proba(testData)

					print 'calculating loss'
					print 'loss:', calc_logloss(prediction, testLabels)
					print 'accuracy', calc_accuracy(prediction, testLabels)


