function removeOutliers
% Normalize HeadeRemoved data using min-max normalization
% For each class in the training set:
% - remove outliers 
% - save to file, 1 file per class 

    pathIn = '..\Data\HeaderRemoved\';
    pathOut = '..\Data\RemoveOutliers\';
    
    outlierOpt = 155;     
%     < 1: exlude n% samples that have highest Mahalanobis Distance
%     example value: 0.95
%     outlierOpt = threshold;     
%     threshold > 1: exclude samples that have Mahalanobis
%     Distance > threshold, example value: 150 or 200, 
%     keyword: Chi square table
%     http://sites.stat.psu.edu/~mga/401/tables/Chi-square-table.pdf
    
    X_training = csvread(strcat(pathIn, 'XTrain.csv'));
    Y_training = csvread(strcat(pathIn, 'YTrain.csv'));

    for i = unique(Y_training)'
        idx = find(Y_training==i);
        N = size(idx, 1);
        fprintf('Class %d:\t%6.0f\t', i, N);
        
        X_i = zscore(X_training(idx, :));
        
        % find the number of components which count 90% of the features
        [~, ~, ~, ~, explained] = pca(X_i);        
        cumSum = cumsum(explained);
        numComponents = find(cumSum<.90*cumSum(end),1,'last');
%         % visuallization
%         plot(cumSum(1:numComponents));
        
        [coeff, score, latent, tsquared, explained] = pca(X_i, 'NumComponents', numComponents);
        residuals = sum((X_i - score*coeff').^2, 2);
        %         % equipvalent to these:
        %         residuals = pcares(X_i, numComponents);
        %         residuals = sum(residuals .^2, 2);

        
        % calculate Mahalanobis Distances to exclude outliers 
        MD = mahal(score, score);
        MD_res = mahal(residuals, residuals);
       
% %         % visuallization
%         f= figure(222);
%         scatter([1:size(MD,1)],MD);
%         title(sprintf('Mahalanobis Distances - class %d', i));
%         saveas(f, strcat(pathOut, sprintf('%d.jpg',i)));
% 
%         f= figure(222);
%         scatter([1:size(MD_res,1)],MD_res);
%         title(sprintf('Mahalanobis Distances Res - class %d', i));
%         saveas(f, strcat(pathOut, sprintf('res%d.jpg',i)));

        if outlierOpt<1            
            % Option 1: take n percentage of samples
            [~,Index] = sort(MD, 'ascend');
            numSamples = floor(outlierOpt * N);
            idxNormals = Index(1:numSamples);
            idxOutliers = Index(numSamples+1:end);
            
            [~,idxRes] = sort(MD_res, 'descend');
            idxResidualOutliers = idxRes(1:(N - numSamples));            
        else
            % Option 2: take samples with Mahalanobis Distance < threshold            
            idxNormals = find(MD<outlierOpt);
            idxOutliers = find(MD>=outlierOpt);
            
            idxResidualOutliers = find(MD_res>=outlierOpt);
        end

        idxNormals(ismember(idxNormals, idxResidualOutliers)) = [];  % remove these from the normal set of the 90% components
        idxOutliers = unique([idxOutliers; idxResidualOutliers]);
        
        X_i = X_training(idx, :);
        X_normals = X_i(idxNormals, :);
        X_outliers = X_i(idxOutliers, :);
        Y_i = Y_training(idx, :);

        fprintf('%6.0f\t%6.0f\n', size(X_normals,1), size(X_outliers,1));

        csvwrite(strcat(pathOut, sprintf('%d.csv', i)), [X_i(idxNormals,:), Y_i(idxNormals,:)]);
        csvwrite(strcat(pathOut, sprintf('o%d.csv', i)), [X_i(idxOutliers,:), Y_i(idxOutliers,:)]);
        
%         % min max normalization
%         csvwrite(strcat(pathOut, sprintf('%d.csv', i)), [bsxfun(@rdivide, bsxfun(@minus, X_normals, mean(X_normals)), (max(X_normals)-min(X_normals))), Y_i(idxNormals,:)]);
%         csvwrite(strcat(pathOut, sprintf('o%d.csv', i)), [bsxfun(@rdivide, bsxfun(@minus, X_outliers, mean(X_outliers)), (max(X_outliers)-min(X_outliers))), Y_i(idxOutliers,:)]);
    end
    
end
