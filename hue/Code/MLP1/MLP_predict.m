% also forwardprop
function [f_a_2, f_a_3, grad_f_a_2, grad_f_a_3] = MLP_predict(model, X)
    a_2 = model.W_1 * X;
    [f_a_2, grad_f_a_2] = sigmoid(a_2);
    a_3 = model.W_2 * f_a_2;
    [f_a_3, grad_f_a_3] = sigmoid(a_3);
end

function [f_a, grad_f_a] = sigmoid(a)
    f_a = 1./(1 + exp(-a));
    grad_f_a = f_a .* (1 - f_a);
end

function f_a = softmax(a)
    a = bsxfun(@minus, a, max(a));
    e_a = exp(a);
    f_a = bsxfun(@rdivide, e_a, sum(e_a, 1));
end

function [grad_f_a, f_a] = tanh(a)
    a = bsxfun(@minus, a, max(a));
    A = exp(a);
    B = exp(-a);
    f_a = bsxfun(@rdivide, (A-B), (A+B));
    grad_f_a = bsxfun(@min, 1, f_a.^2);
end