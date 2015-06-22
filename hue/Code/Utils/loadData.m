function [X_test, Y_test, trainData] = loadData(pathTestSet, pathTrainSet)
    % get test set
    testData = csvread([pathTestSet 'ITest.csv'])';      % 94 x numSamples
    X_test = testData(1:end-1,:);                   % 93 x numSamples
    Y_test = convertLabel(testData(end,:));         %  9 x numSamples
   
    % get train set
    trainData = csvread([pathTrainSet 'ITrain.csv']);      % numSamples x 94
end