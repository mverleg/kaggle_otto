function f = logLoss(Predict, Target)
% evaluate results following this link
% https://www.kaggle.com/c/otto-group-product-classification-challenge/details/evaluation
% Predict = (9 x N)
% Traget = (9 x N)

    Predict = bsxfun(@rdivide, Predict, sum(Predict,1));   % normalize    
    Predict = max(min(Predict,1 - 1.e-15), 1.e-15);
    f = - sum(sum(Target .* log(Predict))) / size(Target,2);
end
