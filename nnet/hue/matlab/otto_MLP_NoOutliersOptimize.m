function otto_MLP_NoOutliers_Optimize
% train on data which have been preprocessed to remove outliers to find the
% best model and test on full data

    clear all;
    clc;
    
    addpath('MLP');

% 	pathData = '..\Data\MinMaxNorm\';
% 	pathData = '..\Data\Classes\';
	pathData = '..\Data\RemoveOutliers\';
% 	pathData = '..\Data\ReducedFeatures\';

	pathFullData = '..\Data\HeaderRemoved\';
    
    numSamples = 1772;
    K = 3;
    W_1 = [];
    W_2 = [];    
    
    rootImages = 'images\';
    if (exist(rootImages, 'dir')==7)
        rmdir(rootImages, 's')
    end
    
    fprintf('Data from %s\nNum samples per class: %d\n', pathData, numSamples);
    bestLossScore = 0.4;
    bestAccuracyScore = 0.75;
    
    for iteration_number = [1000 2000]
        for hidden_unit_number = [25 ]
            for learning_rate = [0.001 0.0001];
                fprintf('Iteration number: %d, Learning rate:%f, Hidden unit number: %d, %d-fold \n', iteration_number, learning_rate, hidden_unit_number, K);
                
                imageFolder = strcat(rootImages, sprintf('I%dH%dA%6f', iteration_number, hidden_unit_number, learning_rate));
                if (exist(imageFolder , 'dir')==0)
                    mkdir(imageFolder);
                end
                h = figure(222);
                
                Accuracy = zeros(K,1);
                LogLoss = zeros(K,1);
                bMin = 0;
                [trainData, testData] = getKFolds(K, numSamples, bMin, pathData);
                
                for k=1:K
                    X_train = trainData{k};
                    X_train = X_train(randperm(size(X_train,1)), :);
                    model = MLP_train(iteration_number, learning_rate, hidden_unit_number, X_train(:, 1:end-1)', convertLabel(X_train(:, end)), W_1, W_2);
                    X_test = testData{k};
                    [~, labels, ~, ~] = MLP_predict(model, X_test(:,1:end-1)');
                    MLP_plotErrors(model);
                    print('-dpng', '-r300', h, sprintf('%s\\k%d', imageFolder, k));
                    Accuracy(k) = MLP_accuracy(labels, convertLabel(X_test(:,end)));
                    LogLoss(k) = logLoss(labels, convertLabel(X_test(:,end)));
                end                
                fprintf('Accuracy \t LogLoss\n');
                disp([Accuracy, LogLoss]);
                fprintf('-------------------------\n');
                meanLoss = mean(LogLoss);
                meanAccuracy = mean(Accuracy);
                disp([meanAccuracy, meanLoss]);
%                 fprintf('%d\t%d\n', meanAccuracy, meanLoss);
                if (meanLoss<bestLossScore)
                    disp('LogLoss improved');
                    bestLossScore = meanLoss;
                    W_1 = model.W_1;
                    W_2 = model.W_2;
                    bestModel = model;
                end
                if (meanAccuracy>bestAccuracyScore)
                    disp('Accuracy improved');
                    bestAccuracyScore = meanAccuracy;
%                     W_1 = model.W_1;
%                     W_2 = model.W_2;
                end
            end
        end
    end
   
    % read full data
    X_data = csvread(strcat(pathFullData, 'XTrain.csv'));
    Y_data = csvread(strcat(pathFullData, 'YTrain.csv'));
    [~, labels, ~, ~] = MLP_predict(bestModel, X_data');
    accuracyScore = MLP_accuracy(labels, convertLabel(Y_data));
    lossScore = logLoss(labels, convertLabel(Y_data));
    fprintf('Result of the best model of the training on full data: Accuracy %d\tLoss %d\n', accuracyScore, lossScore);
    fprintf('Best result of the training: Accuracy %d\tLoss %d\n', bestAccuracyScore, bestLossScore);
    
    csvwrite(sprintf('w1I%dH%dA%5f.csv', bestModel.numIteration, bestModel.numHiddenUnits, bestModel.learningRate), model.W_1);
    csvwrite(sprintf('w2I%dH%dA%5f.csv', bestModel.numIteration, bestModel.numHiddenUnits, bestModel.learningRate), model.W_2);
end