function otto_MLP_NoOutliers
% train on data which have been preprocessed to remove outliers and are normalized
% see minMaxNorm.m

    clear all;
    clc;
    
    addpath('MLP');

% 	pathData = '..\Data\MinMaxNorm\';
% 	pathData = '..\Data\Classes\';
	pathData = '..\Data\RemoveOutliers\';
% 	pathData = '..\Data\ReducedFeatures\';
    
    numSamples = 1772;
    K = 3;
    W_1 = [];
    W_2 = [];    
    
    rootImages = 'images\';
    if (exist(rootImages, 'dir')==7)
        rmdir(rootImages, 's')
    end
    
    fprintf('Data from %s\nNum samples per class: %d\n', pathData, numSamples);
    
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
                disp([mean(Accuracy), mean(LogLoss)]);
            end
        end
    end
end