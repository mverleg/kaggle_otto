function [trainData, testData] = getKFolds(K, numSamples, bMin, pathData)
% create a K-fold data for training and testing
%
% if bMin=1:    take numSamples from each class
% else:         take duplicate samples from each class such that the total
%               samples from each class is numSamples
%
% K				number of folds
% numSamples	number of samples to take from each class, 
% bMin			1, means: numSamples = min (size of classes)
%				0, means: numSamples > min (size of classes)
% pathData		path to the data files, one file per class 
% trainData		cell array contains K cells, each cell contains training data for a fold
% testData		cell array contains K cells, each cell contains the corresponding test data for a fold

    numClasses = 9;

    for k=1:K
        trainData{k} = [];
        testData{k} = [];       
    end
    
    indices = crossvalind('Kfold', numSamples, K);
    
    for i=1:numClasses
        data = csvread(strcat(pathData,sprintf('%d.csv',i)));
        N = size(data,1);
        if bMin
            data = data(randperm(N, numSamples), :);     % select numSamples samples from data
        else
            trainSamples = [];
            numExtra = mod(numSamples, N);
            if (numExtra>0)
                trainSamples = data(randperm(N, numExtra), :);
            end
            for i=1:floor(numSamples/N)
                trainSamples = [trainSamples ; data(randperm(N, N), :)];
            end
            data = trainSamples;
        end
        
        for k =1:K
            testIdx = (indices == k); 
            trainIdx = ~testIdx;
            trainData{k} = [trainData{k} ; data(trainIdx,:)];
            testData{k} = [testData{k} ; data(testIdx, :)];
        end        
    end

end