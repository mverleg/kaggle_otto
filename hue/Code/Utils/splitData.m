function splitData(pathIn, pathOut, K, percent)
% Split full data into two sets: Train and Test where Test set contains percentage number of samples from each class of the whole data set
% repeat the process K times
% Results are K pairs {train set, test set}
% 
% pathIn      path of full data to split
% pathOut     path where K pairs of data are in
% K           number of pairs {train set, test set}
% percent     percentage number of samples from each class in Test set, the rest of samples are in Train set

    if ~exist('pathIn', 'var')
         pathIn = '../Data/ConvertedData/';
    end
    if ~isdir(pathIn)
        error([pathIn ' does not exist. Please check again.']);
    end
    if ~exist('pathOut', 'var')
         pathOut = '../Data/SignificanceTesting/';
    end
    if ~isdir(pathOut)
        mkdir(pathOut);
    end
    if ~exist('K', 'var')
        K = 6;
    end
    if ~exist('percent', 'var')
        percent = 0.1;
    end

    data = csvread([pathIn 'Train.csv']);
	[classSetTmp, ~] = separateClasses(data);
    numClass = size(classSetTmp,1);
    for k=1:K
        testSet = [];
        trainSet = [];
        classSet = cell(numClass, 1);    

        for i=1:numClass
            N = size(classSetTmp{i},1);
            testIdx = randperm(N, floor(percent * N));
            trainIdx = setdiff((1:N), testIdx);
            testSet = [testSet ; classSetTmp{i}(testIdx, :)];
            trainSet = [trainSet ; classSetTmp{i}(trainIdx, :)];
            classSet{i} = classSetTmp{i}(trainIdx, :);       
        end

        csvwrite([pathOut, 'ITest', num2str(k), '.csv'], testSet);
        csvwrite([pathOut, 'ITrain', num2str(k), '.csv'], trainSet);
    end

end