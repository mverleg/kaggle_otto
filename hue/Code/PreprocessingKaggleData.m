function PreprocessingKaggleData
% convert the given Kaggle data to the suitable format, for example
% removing header row, convert label from text to number
% split data to train and evaluate models

    addpath('Utils');

    pathIn = '..\Data\Original\';
    pathOut = '..\Data\ConvertedData1\';
	convertOriginalData(pathIn, pathOut);
    
    splitData(pathOut);
end