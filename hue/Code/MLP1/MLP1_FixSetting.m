function [Accuracy, LogLoss] = MLP1_FixSetting(X_train, Y_train, X_test, Y_test)
    W_1 = [];
    W_2 = [];

    for iteration_number = (2000)
        for hidden_unit_number = (25)
            for learning_rate = (0.001);
%                 fprintf('Iteration number: %d, Learning rate:%f, Hidden unit number: %d \n', iteration_number, learning_rate, hidden_unit_number);            

                model = MLP_train(iteration_number, learning_rate, hidden_unit_number, X_train, Y_train, W_1, W_2);
                [~, labels, ~, ~] = MLP_predict(model, X_test);
                Accuracy = MLP_accuracy(labels, Y_test);
                LogLoss = logLoss(labels, Y_test);
            end
        end
    end

end