% calculate accuracy between predict and target values
function f = MLP_accuracy(Predict, Target)
    % convert to one value, the class label
    [~, assigned_labels] = max(Predict, [], 1);         % Max of column
    [~, true_labels] = max(Target, [], 1);              % Max of column
    f = mean(assigned_labels == true_labels);
end

