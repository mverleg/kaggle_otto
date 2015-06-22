function Otto_MLP_Balance(pathIn, K, balanceFactor)
% test with balance data
% input:
%   pathIn      path where train and test set are stored
%   K           pair of data for significance testing 
%   balanceChoice    0 using oversampling
%                    1 using undersampling

    addpath('MLP1');
    addpath('Utils');
    
    if ~exist('pathIn', 'var')
        pathIn = '../Data/SignificanceTesting/';
    end
    if ~exist('K', 'var')
        K = 6;
    end    
    if ~exist('balanceFactor', 'var')
        balanceFactor = 1.6;              % oversampling
    end
    if (balanceFactor<0)
        balanceFactor = 1;
    end

    fprintf('MLP: ');
    if (balanceFactor>1)
        fprintf('oversampling factor: %2.2f \n', balanceFactor);
    else
        fprintf('undersampling factor: %2.2f \n', balanceFactor);
    end
    
    Accuracy = zeros(K,1);
    LogLoss = zeros(K,1);
    AccuracyS = zeros(K,1);     % for shuffle data
    LogLossS = zeros(K,1);
    fprintf('\tAccuracy\tLogLoss\tAccuracyS\tLogLossS\n');
    
    for k=1:K
        [testData, trainData] = loadSTData(pathIn, k);

        % train data
        [classSet, minClassSize] = separateClasses(trainData);
        if balanceFactor>1
            [selectedData, ~]  = getOversamplingData(classSet, round(balanceFactor * minClassSize));
        else
            [selectedData, ~]  = getUndersamplingData(classSet, round(balanceFactor * minClassSize));
        end
        X_train = selectedData(:,1:end-1)';
        Y_train = convertLabel(selectedData(:,end));

        % test data
        testData = testData';                                       % 94 x numSamples
        X_test = testData(1:end-1,:);                               % 93 x numSamples
        Y_test = convertLabel(testData(end,:));                     %  9 x numSamples

        [Accuracy(k), LogLoss(k)] = MLP1_FixSetting(X_train, Y_train, X_test, Y_test);

        % test with shuffle data
        selectedData = suffleData(selectedData);
        X_train = selectedData(:,1:end-1)';
        Y_train = convertLabel(selectedData(:,end));
        [AccuracyS(k), LogLossS(k)] = MLP1_FixSetting(X_train, Y_train, X_test, Y_test);
        
        fprintf('\t%6.2f\t%6.2f\t%6.2f\t%6.2f\n', Accuracy(k), LogLoss(k), AccuracyS(k), LogLossS(k));
    end

    fprintf('Mean: %6.2f\t%6.2f\t%6.2f\t%6.2f\n', mean(Accuracy), mean(LogLoss), mean(AccuracyS), mean(LogLossS));
    fprintf('Std: %6.2f\t%6.2f\t%6.2f\t%6.2f\n', std(Accuracy), std(LogLoss), std(AccuracyS), std(LogLossS));
  
end