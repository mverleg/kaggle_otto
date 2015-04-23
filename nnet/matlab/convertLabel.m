% =========================================================================
% Convert label vector N x 1 to N x 9
function T = convertLabel(T_training)
    labels= unique(T_training);
    numClasses = size(labels,1);
    T = zeros(size(T_training, 1), numClasses);
    for i=1:numClasses
        T((T_training==i), i) = 1;
    end
    T = T';
end