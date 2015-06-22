function [selectedData, restData]  = getUndersamplingData(classSet, maxSamples)
% from cell array of samples, each cell contains samples of one class, 
% select from each class the same numer of samples 
%
% input:
%   classSet            cell array of samples, each cell contains samples of one class
%   maxSamples          number of samples to take from each class, 
%                       if maxSamples > size of class, all samples of class are
%                       taken
%                       if maxSamples = 0, the function will detect from the
%                       classSet
% output:
%   selectedData        balance data, N x 94
%   restData            the rest of samples

    selectedData = [];
    restData = [];    
    numClasses = size(classSet, 1)
   
    if maxSamples == 0      % find the size of the smallest class        
        for classIdx=1:numClasses
            m = size(classSet{classIdx},1);
            if m < maxSamples
                maxSamples = m;
            end
        end
    end    
    
    for classIdx=1:numClasses
        classData = classSet{classIdx};
        N = size(classData,1);
        numSamples = min(N, maxSamples);
        selectedIdx = randperm(N, numSamples);              % indexes of selected numSamples samples from class
        residualIdx = setdiff((1:N), selectedIdx);          % indexes of the rest from class
        selectedData = [selectedData ; classData(selectedIdx, :)];
        restData = [restData ; classData(residualIdx, :)];                
    end
end