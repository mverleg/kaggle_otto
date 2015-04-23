function getInfo
    X_test = csvread('XTest.csv');    
    X_training = csvread('XTrain.csv');
    T_training = csvread('YTrainOrg.csv');
    
    fprintf('Training samples x features = %d x %d\nNumber of test samples = %d\n', size(X_training), size(X_test,1));
    
    for i = unique(T_training)'
        fprintf('Class %d:\t%6.0f\n', i, sum(T_training==i));
    end

    if sum(sum(isnan(X_training)))>0
        disp('X_training has missing value');
    end
    
    if sum(sum(isnan(X_test)))>0
        disp('X_test has missing value');
    end
end
