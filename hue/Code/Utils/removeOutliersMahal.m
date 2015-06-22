function [noOutlierSet, outlierSet] = removeOutliersMahal(outlierOpt, data, verbose)
% Remove outlier from each class of data, using Mahal distance
%
% Output:
%   noOutlierSet    set contains no outliers, N x D
%   outlierSet      set contains outliers, N x D
% Input:
%   outlierOpt < 1: percentage of samples to be excluded that have highest Mahalanobis distance, example value: 0.95
%	outlierOpt > 1: threshold to exclude samples that have Mahalanobis Distance > threshold, example value: 150 or 200, 
%                   keyword: Chi square table
%                   http://sites.stat.psu.edu/~mga/401/tables/Chi-square-table.pdf
%                   these two values below are set based on prior knowledge, these could
%                   also be found by looping through classes
%   data:           given data to be processed
  
    
%     if (outlierOpt>1)
%         fprintf('Remove outlier using threshold = %d\n', outlierOpt);
%     else
%         fprintf('Remove outlier using cutoff percentage = %4.2f\n', outlierOpt);
%     end

    if ~exist('verbose', 'var')
        verbose = 0;
    end

    [classSet, ~] = separateClasses(data);
    numClasses = length(classSet);
    
    outlierSet = [];  % to hold outliers of all classes 
    noOutlierSet = [];
    
    for i=1:numClasses
        classData = classSet{i};
        N = size(classData, 1);
        if verbose
            fprintf('Class %d:\t%6.0f\t', i, N);        
        end
        X_i = classData(:, 1:end-1);

        % calculate Mahalanobis Distances to exclude outliers 
        MD = mahal(X_i, X_i);
       
%         % visuallization
%         f= figure(222);
%         scatter([1:size(MD,1)],MD);
%         title(sprintf('Mahalanobis Distances - class %d', i));
%         saveas(f, [pathOut, sprintf('%d.jpg',i)]);

        if outlierOpt>1
            % take samples with Mahalanobis Distance < threshold            
            idxNormals = find(MD<outlierOpt);
            idxOutliers = find(MD>outlierOpt);
        else
            % take percentage of samples
            [~,Index] = sort(MD, 'ascend');
            numSamples = floor(outlierOpt * N);
            idxNormals = Index(1:numSamples);
            idxOutliers = Index(numSamples+1:end);            
        end
        
        noOutlierSet = [noOutlierSet ; classData(idxNormals,:)];
        outlierSet = [outlierSet ; classData(idxOutliers,:)];
        if verbose
            fprintf('%6.0f\t%6.0f\n', size(idxNormals,1), size(idxOutliers,1));
        end
    end
end
