function Otto_MLP
    addpath('MLP');
    
    % load data
    X_training = csvread('XTrain.csv');
    T_training = csvread('YTrain.csv');
    X_test = csvread('XTest.csv');

    % convert to make it suitable for NN code
    X_training = X_training';                   % numFeatures x numSamples
    X_test = X_test';                           % numFeatures x numSamples
    
    % initialize parameters
    iteration_number = 20000;
    hidden_unit_number = 25;
    learning_rate = 0.001;

    % cross validation
    W_1 = [];
    W_2 = [];

    fprintf('Iteration number: %d, Learning rate:%f, Hidden unit number: %d \n', iteration_number, learning_rate, hidden_unit_number);            


    model = MLP_train(iteration_number, learning_rate, hidden_unit_number, X_training, T_training, W_1, W_2);
    [~, labels, ~, ~] = MLP_predict(model, X_test);
    h = figure(222);
    MLP_plotErrors(model);        

    % test on X_training to see how good it is
    [~, labels, ~, ~] = MLP_predict(model, X_training);
    fprintf('Accuracy on X_training %5f\n', MLP_accuracy(labels, T_training));

    csvwrite(sprintf('w1I%dH%dA%5f.csv', iteration_number, hidden_unit_number, learning_rate), model.W_1);
    csvwrite(sprintf('w2I%dH%dA%5f.csv', iteration_number, hidden_unit_number, learning_rate), model.W_2);

    % make submission
    labels = round(labels', 4);
    results = array2table([[1:size(labels,1)]' labels], 'VariableNames',{'id' 'Class_1' 'Class_2' 'Class_3' 'Class_4' 'Class_5' 'Class_6' 'Class_7' 'Class_8' 'Class_9'});
    writetable(results, 'result.csv', 'Delimiter', ',');
    
end