function [classSet, minSize] = separateClasses(data, verbose)
% return data separated into classes
% inout:
%   data          N x 94
%   classSet      cell array of samples from data, where each cell contains
%                 samples of one class
%   minSize       size of the smallest class in classSet

    if ~exist('verbose', 'var')
        verbose = 0;
    end
        
    numClass = 9;
    minSize = size(data,1);    
    Y_training = data(:, end);
    classSet = cell(numClass, 1);
    for i = 1:numClass
        idx = find(Y_training==i);
        M = size(idx, 1);
        classSet{i} = data(idx, :);
        if minSize > M
            minSize = M;
        end
        if verbose
            fprintf('Class %d:\t%6.0f\n', i, M);
        end
    end
end
