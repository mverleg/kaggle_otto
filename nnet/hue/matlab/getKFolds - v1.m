function [trainData, testData] = getKFolds(K, minSamples, maxSamples, pathData)
% create a K-fold data for training and testing
% K			number of folds
% numClasses
% minSamples	number of samples to take from each class, <= min (size of classes)
% pathData	path to the data files, one file per class 
% trainData	cell array contains K cells, each cell contains training data for a fold
% testData	cell array contains K cells, each cell contains the corresponding test data for a fold

% 	pathData = '..\Data\MinMaxNorm\';
%     numSamples = 1767;           % min (number of samples of class)
    
    numClasses = 9;

    for k=1:K
        trainData{k} = [];
        testData{k} = [];       
    end
    
    if minSamples > 0
        numSamples = minSamples;
    else
        numSamples = maxSamples;
    end
    
    indices = crossvalind('Kfold', minSamples, K);
    
    for i=1:numClasses
        data = csvread(strcat(pathData,sprintf('%d.csv',i)));
        data = data(randperm(size(data,1), minSamples), :);     % select N samples from data
        
        for k =1:K
            testIdx = (indices == k); 
            trainIdx = ~testIdx;
            trainData{k} = [trainData{k} ; data(trainIdx,:)];
            testData{k} = [testData{k} ; data(testIdx, :)];
        end        
    end

end