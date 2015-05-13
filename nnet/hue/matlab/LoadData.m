function [X_training, T_training, X_test] = LoadData()
    X_training = csvread('XTrain.csv');
    X_test = csvread('XTest.csv');

    X_training = X_training';                   % numFeatures x numSamples
    X_test = X_test';                           % numFeatures x numSamples
	T_training = csvread('YTrain.csv');         % converted already
end