function Otto_MLP
    addpath('MLP');
    
    % load data
    [X_training, T_training, X_test] = LoadData();
%     X_training = csvread('XTrain.csv');
%     X_test = csvread('XTest.csv');
% 
%     X_training = X_training';                   % numFeatures x numSamples
%     X_test = X_test';                           % numFeatures x numSamples
% 	T_training = csvread('YTrain.csv');         % converted already

    
    % initialize parameters
%     iteration_number = 200;
%     hidden_unit_number = 25;
%     learning_rate = 0.0001;

    % cross validation
    K = 3;
    N = size(X_training, 2);

    rootImages = strcat(pwd, '\\images\\');
    if (exist(rootImages, 'dir')==7)
        rmdir(rootImages, 's')
    end

    W_1 = [];
    W_2 = [];
% 	W_1 = csvread('w1I2000H25A0.000001.csv');
% 	W_2 = csvread('w2I2000H25A0.000001.csv');

    maxScore = 0.80;
%     lastScore = 0;
    
    for iteration_number = [1000]
        for hidden_unit_number = [25 ]
            for learning_rate = [0.001];
%     for iteration_number = [200]
%         for hidden_unit_number = [25]
%             for learning_rate = [0.0001];
                fprintf('Iteration number: %d, Learning rate:%f, Hidden unit number: %d \n', iteration_number, learning_rate, hidden_unit_number);            
                
                indices = crossvalind('Kfold', N, K);    
                Accuracy = zeros(K,1);
                LogLoss = zeros(K,1);

                imageFolder = strcat(rootImages, sprintf('I%dH%dA%6f', iteration_number, hidden_unit_number, learning_rate));
                if (exist(imageFolder , 'dir')==0)
                    mkdir(imageFolder);
                end
                h = figure(222);
                    
                for k =1:K
                    fprintf ('Fold %d\t', k);
                    test = (indices == k); 
                    train = ~test;
                    model = MLP_train(iteration_number, learning_rate, hidden_unit_number, X_training(:,train), T_training(:,train), W_1, W_2);
                    [~, labels, ~, ~] = MLP_predict(model, X_training(:,test));

                    MLP_plotErrors(model);        
                    print('-dpng', '-r300', h, sprintf('%s\\k%d', imageFolder, k));

                    Accuracy(k) = MLP_accuracy(labels, T_training(:,test));
                    LogLoss(k) = logLoss(labels, T_training(:,test));
                end
                fprintf('Accuracy \t LogLoss\n');
                disp([Accuracy, LogLoss]);
                fprintf('-------------------------\n');
                disp([mean(Accuracy), mean(LogLoss)]);

                score = mean(Accuracy);
%                 fprintf('Mean accuracy %f\n', score);                
                if (score>maxScore)
                    maxScore = score;
                    W_1 = model.W_1;
                    W_2 = model.W_2;
                    csvwrite(sprintf('w1I%dH%dA%5f.csv', iteration_number, hidden_unit_number, learning_rate), model.W_1);
                    csvwrite(sprintf('w2I%dH%dA%5f.csv', iteration_number, hidden_unit_number, learning_rate), model.W_2);
                end
            end
        end
    end

%     csvwrite('result.csv', f_a_3);
%     csvwrite('w1.csv', model.W_1);
%     csvwrite('w2.csv', model.W_2);

end


