function Otto_TreeBagger_Final()
% create submission

    addpath('Utils');       % server

    % create pool for parallel
    numWorkers = 30;
	poolObj = createPool(numWorkers); % Invoke workers    

    pathIn = '../Data/ConvertedData/Test.csv';   % server
    testData = csvread(pathIn);    

    pathIn = '../Data/ConvertedData/Train.csv';   % server
    trainData = csvread(pathIn);
    
    options = statset('UseParallel',1);
    nTree = 600;
    model = TreeBagger(nTree, trainData(:,1:end-1), trainData(:,end), 'Options', options);
    [~,score] = predict(model, testData);    

    pathOut = 'Result/';
    if ~isdir(pathOut)
        mkdir(pathOut);
    end
    save([pathOut 'model.mat'], 'model');
    createSubmission(score, pathOut);
   
    delete(poolObj);
end
