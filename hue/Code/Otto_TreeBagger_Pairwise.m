function Otto_TreeBagger_Pairwise()
% pairwise classification

    addpath('Utils');

    if ~exist('pathIn', 'var')
        pathIn = '../Data/SignificanceTesting/';
    end
    if ~exist('pathOut', 'var')
        pathOut = '../Result/';
    end
    if ~isdir(pathOut)
        mkdir(pathOut);
    end
    
    % create pool for parallel
    numWorkers = 36;
	poolObj = createPool(numWorkers); % Invoke workers    
    
    nTree = 600;
    fprintf('TreeBagger Pairwise numTree: %d\n', nTree);
    
    k = 1;
    [testData, trainData] = loadSTData(pathIn, k);
    [classTestSet, ~] = separateClasses(testData);
    [classTrainSet, ~] = separateClasses(trainData);
        
    M = 9;  % number of classes
    idx1 = zeros(M*(M-1)/2,1);
    idx2 = zeros(M*(M-1)/2,1);
    i = 1;
    for cIdx1 = 1:M-1
        for cIdx2=cIdx1+1:M
            idx1(i) = cIdx1;
            idx2(i) = cIdx2;
            i = i+1;
        end
    end
    
    Models = cell(size(idx1));      % cell matrix to keep tract of pairwise models
   
    
    % training to get models
	tic();
    parfor i=1:size(idx1,1)
        Models{i} = trainPairwiseTreeBagger(nTree, [classTrainSet{idx1(i)} ; classTrainSet{idx2(i)}],...
            [classTestSet{idx1(i)} ; classTestSet{idx2(i)}], idx1(i), idx2(i), pathOut);
    end
    fprintf('Training Time %d\n', toc());

    
    % predict samples
    tic();
    X = testData(:, 1:end-1);
    parfor i=1:size(idx1,1)
        predictOnTestSet(Models{i}, X, idx1(i), idx2(i), pathOut);
    end
    fprintf('Test Time %d\n', toc());
    
    
    % gathering pairwise log into one file
    getPairwiseResult(pathOut);
    
    
    % gathering pairwise results
    N = size(testData,1);
    P = zeros(N,M,M);    
    for cIdx1 = 1:M-1
        for cIdx2=cIdx1+1:M
            labels = csvread([pathOut num2str(cIdx1) 'v' num2str(cIdx2) '.csv']);
            P(:, cIdx1, cIdx2) = labels;
            P(:, cIdx2, cIdx1) = labels;
        end
    end
    
    [modeP, fP] = mode(P, 3);
    [~,mIdx] = max(fP, [], 2);
    labels = modeP(sub2ind([N,M], (1:N)', mIdx));
    
    fprintf('Test pairwise models: Accuracy %6.4f\n', mean(labels == testData(:,end)));    

% %     % test values
%     for i=1:N
%         squeeze(P(i,:,:))
%     end    

    delete(poolObj);    
    
end

function model = trainPairwiseTreeBagger(nTree, trainData, testData, idx1, idx2, pathOut)
% trainSet:         train data from two classes
% testData:         test data from two classes 
% idx1:             index of class 1
% idx2:             index of class 2
% pahtOut:          path to dumpt result

    options = statset('UseParallel',1);
    model = TreeBagger(nTree, trainData(:,1:end-1), trainData(:,end), 'Options', options);

    [labels,~] = predict(model, testData(:,1:end-1));    
    labels = str2num(cell2mat(labels));

    fileID = fopen(sprintf('%s%dv%d.txt', pathOut, idx1, idx2),'w');    
    fprintf(fileID, '%6.4f', mean(labels == testData(:,end)));
    fclose(fileID);
end

function labels = predictOnTestSet(model, X, idx1, idx2, pathOut)
    [labels,~] = predict(model, X);
    labels = str2num(cell2mat(labels));
    csvwrite(sprintf('%s%dv%d.csv', pathOut, idx1, idx2), labels);
end

function getPairwiseResult(pathIn)

    if ~exist('pathIn', 'var')
        pathIn = '../Result/';
    end

    M = 9;
    data = zeros(M);
    for cIdx1 = 1:M-1
        for cIdx2=cIdx1+1:M
            fileID = fopen([pathIn num2str(cIdx1) 'v' num2str(cIdx2) '.txt']);
            data(cIdx1, cIdx2) = fscanf(fileID,'%f');
            fclose(fileID);
        end
    end

    csvwrite([pathIn 'combine.csv'], data);
end
