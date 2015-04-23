function [trainData, testData] = getKFolds(K, maxSamples, bMin, pathData)
% create a K-fold data for training and testing
%
% K				number of folds
% maxSamples	number of samples to take from each class, 
%               if maxSamples > size of class, all samples of class are
%               taken
% bMin			not use
% pathData		path to the data files, one file per class 
% trainData		cell array contains K cells, each cell contains training data for a fold
% testData		cell array contains K cells, each cell contains the corresponding test data for a fold

    numClasses = 9;

    for k=1:K
        trainData{k} = [];
        testData{k} = [];       
    end
    
    for i=1:numClasses
        data = csvread(strcat(pathData,sprintf('%d.csv',i)));
        N = size(data,1);
        numSamples = min(N, maxSamples);
        data = data(randperm(N, numSamples), :);     % select numSamples samples from data
        
        indices = crossvalind('Kfold', numSamples, K);
        for k =1:K
            testIdx = (indices == k); 
            trainIdx = ~testIdx;
            trainData{k} = [trainData{k} ; data(trainIdx,:)];
            testData{k} = [testData{k} ; data(testIdx, :)];
        end
    end

end