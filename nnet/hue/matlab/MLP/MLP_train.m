% ========================================================================
% 2 layer perceptron
% Hue Dang
% ========================================================================

function model = MLP_train(iteration_number, learning_rate, hidden_unit_number, X_training, T_training, W1, W2)
% 2 layer perceptron
% iteration_number        example value 500
% learning_rate           example value 0.001
% hidden_unit_number      example value 25     
% X_training              training data,matrix numFeatures x numSamples
% T_training              label of training data, matrix numClasses x numSamples, classes are in format: [1 0 0]' = class 1
% W1                      weight matrix hiddenUnits x numFeatures 
% W2                      weight matrix numClasses x hiddenUnits

    % initialize Weights
    input_unit_number = size(X_training, 1);
    output_unit_number = size(T_training, 1);

    definedW = 0;
    if (exist('W1', 'var') && (exist('W2', 'var')))
        if (isequal(size(W1),[hidden_unit_number, input_unit_number]) && isequal(size(W2), [output_unit_number, hidden_unit_number]))
            definedW = 1;
            W_1 = W1;
            W_2 = W2;
        end
    end
    if (~definedW)
        % NxM random values in range [begin, end]: (end - begin) * rand(N,M) + begin;
        endValue = sqrt(6/(hidden_unit_number + input_unit_number));
        W_1 = 2*endValue*rand(hidden_unit_number, input_unit_number) - endValue;

        endValue = sqrt(6/(hidden_unit_number + output_unit_number));
        W_2 = 2*endValue*rand(output_unit_number, hidden_unit_number) - endValue;
    end
    
    lamda = 1.9;
    momentum = 0.5;
    velocityW1 = zeros(size(W_1));
    velocityW2 = zeros(size(W_2));
    
    % Training     
    E = zeros(1, iteration_number);                     % variable to keep track of error during the training
%     tic();                                              % time at begin 
    for iteration = 1 : iteration_number        
        % disp(['Iteration: ' num2str(iteration) ' / ' num2str(iteration_number)]); 
        [E_W, grad_E_W_1, grad_E_W_2] = error(W_1, W_2, X_training, T_training);  
        % disp(['Error: ' num2str(E_W)]);                                           
        E(iteration) = E_W;                             % keep tract of error

        % increase momentum after 20 iterations
        if iteration==20
            momentum = 0.9;
        end
        
%         W_1 = W_1 - learning_rate * grad_E_W_1;
%         W_2 = W_2 - learning_rate * grad_E_W_2;

        W_1 = W_1 - learning_rate * grad_E_W_1 - lamda * learning_rate * W_1;
        W_2 = W_2 - learning_rate * grad_E_W_2 - lamda * learning_rate * W_2;

%         velocityW1 = (momentum .* velocityW1) - (learning_rate .* grad_E_W_1);
%         W_1 = W_1 - velocityW1;
%         velocityW2 = (momentum .* velocityW2) - (learning_rate .* grad_E_W_2);
%         W_2 = W_2 - velocityW2;

%         if mod(iteration, 100)==0
%             disp([max(W_1(:)) max(W_2(:)); min(W_1(:)) min(W_2(:))]);
%         end
    end                                                                           
%     disp(['Running time: ' num2str(toc()) ' seconds']); % end time

    model = modelInit(W_1, W_2, iteration_number, learning_rate, hidden_unit_number, E);

end

function [grad_E_W_1, grad_E_W_2] = backprop(f_a_2, f_a_3, grad_f_a_2, grad_f_a_3, T, W_2, X)
    delta_3 = grad_f_a_3 .* (f_a_3 - T);        % Bishop 19
    delta_2 = grad_f_a_2 .* (W_2' * delta_3);   % Bishop 25
    grad_E_W_1 = delta_2 * X';                  % Bishop 23
    grad_E_W_2 = delta_3 * f_a_2';              % Bishop 18
end

function [E_W, grad_E_W_1, grad_E_W_2] = error(W_1, W_2, X, T)
    model = modelInit(W_1, W_2);
	[f_a_2, f_a_3, grad_f_a_2, grad_f_a_3] = MLP_predict(model, X);
    E_W = sum(sum((f_a_3 - T).^2, 1), 2)/2;       % Bishop 6
	[grad_E_W_1, grad_E_W_2] = backprop(f_a_2, f_a_3, grad_f_a_2, grad_f_a_3, T, W_2, X);
end
