function Otto_MLP1
% train and test MLP using 3-fold cross-validation

    addpath('MLP1');
    addpath('Utils');
    pathIn = '..\Data\ConvertedData\';
    
    % load data
    trainData = csvread([pathIn 'Train.csv'])';     % 94 x numSamples
    X_training = trainData(1:end-1,:);              % 93 x numSamples
    Y_training = convertLabel(trainData(end,:));    %  9 x numSamples
        
    % cross validation
    K = 3;
    N = size(X_training, 2);

    W_1 = [];
    W_2 = [];

    % load weights from previous training
% 	W_1 = csvread('w1I2000H25A0.000001.csv');
% 	W_2 = csvread('w2I2000H25A0.000001.csv');

    maxScore = 0.80;

    numTrials = 3;      % run several times
    for t=1:numTrials
        for iteration_number = [2000]
            for hidden_unit_number = [25 ]
                for learning_rate = [0.001];
                    fprintf('Iteration number: %d, Learning rate:%f, Hidden unit number: %d \n', iteration_number, learning_rate, hidden_unit_number);            

                    indices = crossvalind('Kfold', N, K);    
                    Accuracy = zeros(K,1);
                    LogLoss = zeros(K,1);

                    for k =1:K
                        fprintf ('Fold %d\t', k);
                        test = (indices == k); 
                        train = ~test;

                        model = MLP_train(iteration_number, learning_rate, hidden_unit_number, X_training(:,train), Y_training(:,train), W_1, W_2);
                        [~, labels, ~, ~] = MLP_predict(model, X_training(:,test));

                        Accuracy(k) = MLP_accuracy(labels, Y_training(:,test));
                        LogLoss(k) = logLoss(labels, Y_training(:,test));
                    end
                    fprintf('Accuracy \t LogLoss\n');
                    disp([Accuracy, LogLoss]);
                    fprintf('-------------------------\n');
                    disp([mean(Accuracy), mean(LogLoss)]);

                    score = mean(Accuracy);
                    if (score>maxScore)
                        maxScore = score;
%                         W_1 = model.W_1;
%                         W_2 = model.W_2;
                        csvwrite(sprintf('w1I%dH%dA%5f.csv', iteration_number, hidden_unit_number, learning_rate), model.W_1);
                        csvwrite(sprintf('w2I%dH%dA%5f.csv', iteration_number, hidden_unit_number, learning_rate), model.W_2);
                    end
                end
            end
        end
    end

end