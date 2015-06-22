function selectFeaturesLDA23()
% select best features to classify class 2 and 3

    % server
     addpath('Utils');
     pathIn = '../Data/IClasses/';

    [classSet, minClassSize] = getClassesFromFile(pathIn)
    data = [classSet{2} ; classSet{3}];
    X = data(:, 1:end-1);
    Y = data(:, end);

    opts = statset('display','iter');
    [inmodel, history] = sequentialfs(@LDA,X,Y, 'options',opts);
    csvwrite('featuresLDA.csv', inmodel);
end

function f=classify(Xtrain, Ytrain, Xtest, Ytest)
    f = 1;
    fID = fopen('logSelectFeatures.txt', 'a');
    options = statset('UseParallel',1);
    model = TreeBagger(600, Xtrain, Ytrain, 'Options', options);
    [labels,~] = predict(model, Xtest);
    labels = str2num(cell2mat(labels));
    f = sum(Ytest~=labels);
    fprintf(fID, '%d\n', f);
    fclose(fID);
end

function f=LDA(Xtrain, Ytrain, Xtest, Ytest)
%     f = 1;
%     fID = fopen('logSelectFeatures.txt', 'a');
%     options = statset('UseParallel',1);
    model = fitcdiscr(Xtrain, Ytrain, 'discrimType','pseudoLinear');
    labels = predict(model, Xtest);
    f = sum(Ytest~=labels);
%     fprintf(fID, '%d\n', f);
%     fclose(fID);
end
