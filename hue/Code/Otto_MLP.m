function Otto_MLP(pathIn, K)
% train MLP on train set and evaluate it on test set
% input:
%   pathIn      path where train and test set are stored
%   K           pair of data for significance testing 

    addpath('MLP1');
    addpath('Utils');
    
    if ~exist('pathIn', 'var')
        pathIn = '../Data/SignificanceTesting/';
    end
    if ~exist('K', 'var')
        K = 6;
    end    
    
    fprintf('MLP \n');
    Accuracy = zeros(K,1);
    LogLoss = zeros(K,1);
    AccuracyS = zeros(K,1);     % for shuffle data
    LogLossS = zeros(K,1);
    fprintf('\tAccuracy\tLogLoss\tAccuracyS\tLogLossS\n');
    for k=1:K
        [testData, trainData] = loadSTData(pathIn, k);
        trainData = trainData';                         % 94 x numSamples
        X_train = trainData(1:end-1,:);                 % 93 x numSamples
        Y_train = convertLabel(trainData(end,:));       %  9 x numSamples
        testData = testData';                                       % 94 x numSamples
        X_test = testData(1:end-1,:);                               % 93 x numSamples
        Y_test = convertLabel(testData(end,:));                     %  9 x numSamples

        [Accuracy(k), LogLoss(k)] = MLP1_FixSetting(X_train, Y_train, X_test, Y_test);

        % Test with shuffle train data
        trainData = suffleData(trainData')';    
        X_train = trainData(1:end-1,:);                 % 93 x numSamples
        Y_train = convertLabel(trainData(end,:));       %  9 x numSamples

        [AccuracyS(k), LogLossS(k)] = MLP1_FixSetting(X_train, Y_train, X_test, Y_test);
        
        fprintf('\t%6.2f\t%6.2f\t%6.2f\t%6.2f\n', Accuracy(k), LogLoss(k), AccuracyS(k), LogLossS(k));
    end

    fprintf('Mean: %6.2f\t%6.2f\t%6.2f\t%6.2f\n', mean(Accuracy), mean(LogLoss), mean(AccuracyS), mean(LogLossS));
    fprintf('Std: %6.2f\t%6.2f\t%6.2f\t%6.2f\n', std(Accuracy), std(LogLoss), std(AccuracyS), std(LogLossS));
    
    
end
