function [testData, trainData] = loadSTData(pathIn, k)
% load pair k of the significance testing
% input:
%     pathIn      path to the data
%     k           pair k of data in the significance testing
% output:
%     testData    test data
%     trainData   data for train
    
    % get test set
    testData = csvread([pathIn 'ITest' num2str(k) '.csv']);     % numSamples x 94
   
    % get train set
    trainData = csvread([pathIn 'ITrain' num2str(k) '.csv']);           % numSamples x 94
end