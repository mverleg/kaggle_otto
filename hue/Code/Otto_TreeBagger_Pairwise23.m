function Otto_TreeBagger_Pairwise23()
% TreeBaggers pairwise classification for class 2 and 3

    addpath('Utils');       % server

    if ~exist('pathIn', 'var')
        pathIn = '../Data/SignificanceTesting/';
    end

    % create pool for parallel
    numWorkers = 30;
	poolObj = createPool(numWorkers); % Invoke workers    
    nTree = 600;
    fprintf('TreeBagger Pairwise 2v3 numTree: %d\n', nTree);

    k = 1;
    [testData, trainData] = loadSTData(pathIn, k);
    [classTestSet, ~] = separateClasses(testData);
    [classTrainSet, ~] = separateClasses(trainData);   
    
    class1 = 2;
    class2 = 3;
    trainData = [classTrainSet{class1} ; classTrainSet{class2}];
    testData = [classTestSet{class1} ; classTestSet{class2}];
    C = [
        0 4;
        6 0;
        ];

    % training to get models
    options = statset('UseParallel',1);
    model = TreeBagger(nTree, trainData(:,1:end-1), trainData(:,end), 'Options', options, 'Cost', C);
    [labels,~] = predict(model, testData(:,1:end-1));    
    labels = str2num(cell2mat(labels));
    fprintf('Accuracy %6.4f\n', mean(labels == testData(:,end)));
    confusionmat(testData(:,end), labels)
    
    delete(poolObj);    
    
end