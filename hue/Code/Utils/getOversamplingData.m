function [selectedData, restData]  = getOversamplingData(classSet, numSamples)
% from cell array of samples, each cell contains samples of one class, 
% select from each class the same numer of samples 
% samples from small classes are duplicated to fulfill the required number
%
% input:
%   classSet            cell array of samples, each cell contains samples of one class
%   numSamples          number of samples to take from each class, 
% output:
%   selectedData        balance data, N x 94
%   restData            the rest of samples


    numClasses = size(classSet,1);
    selectedData = [];
    restData = [];
    
    if numSamples <= 0      
        error('maxSamples should >= 0');
    end

    for classIdx=1:numClasses
        classData = classSet{classIdx};
        N = size(classData,1);
        permSamples = numSamples;
        if N<numSamples
            % replicate
            t = floor(numSamples/N);
            for i=1:t
                selectedData = [selectedData ; classData];
            end
            permSamples = mod(numSamples,N);
        end
        
        if permSamples>0
            selectedIdx = randperm(N, permSamples);              % indexes of selected numSamples samples from class
            selectedData = [selectedData ; classData(selectedIdx,:)];
        end
        
        if N>numSamples
            residualIdx = setdiff((1:N), selectedIdx);    % indexes of the rest from class        
            restData = [restData ; classData(residualIdx, :)];        
        end        
    end

end