function trainSet = convertOriginalData(pathIn, pathOut)
% convert the given data from Kaggle to format that is possible for training
% - removing header row and index column
% - convert class column from text to number
% save to Train.csv and Test.csv, corresponding to train set and test set
% Train.csv : (N x 94)
% Test.csv : (N x 93)
%
% input:
%   pathIn:     path where the given Kaggle data is
%   pathOut:    path to save the converted data in
% output: 
%   trainSet:   the contain of Train.csv as matrix N x 94

    if ~exist('pathIn', 'var')
        pathIn = '..\..\Data\Original\';
    end
    if ~isdir(pathIn)
        error(sprintf('Input path %s does not exist. Please check again.', pathIn));
    end

    if ~exist('pathOut', 'var')
        pathOut = '..\..\Data\ConvertedData\';
    end
    if ~isdir(pathOut)
        mkdir(pathOut);
    end


    % ========================================================
    % TEST SET

    [M, ~]= xlsread([pathIn, 'test.csv']);

    % ignore first column
    csvwrite([pathOut, 'Test.csv'], M(:, 2:end));

    
    % ========================================================
    % TRAINING SET
    [M, T]= xlsread([pathIn, 'train.csv']);

    % skip first row, get last column
    T = T(2:end,end);

    % convert text to number
    T2 = zeros(size(T));
    Class = unique(T);
    for i=1:size(Class,1)
        T2(ismember(T,Class(i))) = i;
    end

    trainSet = [M(:, 2:end), T2];

    csvwrite([pathOut, 'Train.csv'], trainSet);
end