function C = getCostFunction(data)
% Given the data, a cost function is returned based on the paper 'MetaCost: a general method for making classifiers cost-sensitive',
% Domingos, P. - Proceedings of the fifth ACM SIGKDD international. 1999
% it is required that the data reflects the distribution of classes in both train and test sets
% Input:
%     data:   data to get the information to build the cost matrix, N x D
% Output:
%     C:      cost matrix, M x M, where M is the number of classes
    
    [classSet, ~] = separateClasses(data);
    M = length(classSet);
    P = zeros(M,1);
    for i=1:M
        P(i) = length(classSet{i});
    end
    C = zeros(M);
    for i=1:M
        for j=1:M
%             C(i,j) = 1000 * P(i)/P(j);
            value = round(2000 * P(i)/P(j)) + 1;
            C(i,j) = randi(value)-1;
        end
    end
    C(1:(size(C,1)+1):end) = 0;             % C(i,i) = 0
end