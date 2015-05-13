function convertOriginalData
% ==========================================================================
% convert the given data from Kaggle to format that is possible for training
% - removing header row and index column
% - convert class column from text to number
% ==========================================================================

    pathIn = '..\Data\Original\';
    pathOut = '..\Data\HeaderRemoved\';

    % ========================================================
    % TRAINING SET
    [M T]= xlsread(strcat(pathIn, 'train.csv'));


    % ========================================================
    % Save X_training
    % ignore first column
    csvwrite(strcat(pathOut,'XTrain.csv'), M(:, 2:end));
    clear M;


    % ========================================================
    % Save Y_training

    % skip first row, get last column
    T = T(2:end,end);

    % convert text to number
    T2 = zeros(size(T));
    Class = unique(T);
    for i=1:size(Class,1)
        T2(ismember(T,Class(i))) = i;
    end

    % % test 
    % for i=1:size(Class,1)    
    %     idx1 = ismember(T1,Class(i));
    %     idx2 = (T2==i);
    %     fprintf('%d %d\n', i, isequal(idx1, idx2));    
    % end
    % clear idx1 idx2;

    csvwrite(strcat(pathOut, 'YTrain.csv'), T2);
    clear T T2;


    % ========================================================
    % TEST SET

    [M ~]= xlsread(strcat(pathIn, 'test.csv'));

    % ignore first column
    csvwrite(strcat(pathOut, 'XTest.csv'), M(:, 2:end));
    clear M;
end