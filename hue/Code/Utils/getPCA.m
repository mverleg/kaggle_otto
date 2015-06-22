function [newTrain, newTest] = getPCA(trainData, testData, percent)
% transform the given data to another dimension using PCA
% Input:
%     data:       data to be transformed, N x D
%     percent:    percentage of the information to count the number of components
% Output:
%     newTrain:   transformed of trainData, N x D
%     newTest:    transformed of testData, N x D

    if ~exist('percent', 'var')
        percent = 0.9;
    end
    
    X = trainData(:, 1:end-1);
    
    % find the number of components which count 90% of the features
    [coeff, score, latent, tsquared, explained] = pca(trainData(:, 1:end-1));
    cumSum = cumsum(explained);
    numComponents = find(cumSum<percent*cumSum(end),1,'last');
    fprintf('numComponents: %d\n', numComponents);
    newTrain = [score(:,1:numComponents) trainData(:,end)];
    
    [coeff, score, latent, tsquared, explained] = pca(testData(:,1:end-1), 'NumComponents', numComponents);
    newTest = [score testData(:,end)];
end