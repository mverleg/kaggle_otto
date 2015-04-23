function separateClasses
% From HeadeRemoved data to files
% Each class in one file: [data, class]
% class is a number: 1..9

    pathIn = '..\Data\HeaderRemoved\';
    pathOut = '..\Data\Classes\';
  
    X_training = csvread(strcat(pathIn, 'XTrain.csv'));
    Y_training = csvread(strcat(pathIn, 'YTrain.csv'));

    for i = unique(Y_training)'
        idx = find(Y_training==i);
        fprintf('Class %d:\t%6.0f\n', i, size(idx, 1));
        csvwrite(strcat(pathOut, sprintf('%d.csv', i)), [X_training(idx, :), Y_training(idx,:)]);
    end
    
end
