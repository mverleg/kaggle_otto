function createSubmission(score, pathOut)
    results = array2table([(1:size(score,1))' score], 'VariableNames',{'id' 'Class_1' 'Class_2' 'Class_3' 'Class_4' 'Class_5' 'Class_6' 'Class_7' 'Class_8' 'Class_9'});
    writetable(results, [pathOut 'submission.csv'], 'Delimiter', ',');    
end