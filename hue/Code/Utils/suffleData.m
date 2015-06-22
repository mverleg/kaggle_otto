function sX = suffleData(X)
% changing order of samples in X randomly
% X is required to be in format samples x features (N x D)

    sX = X(randperm(size(X,1)), :);
end