function model = modelInit(W_1, W_2, iterations, learningRate, numHiddenUnit, Errors)
    model.W_1 = W_1;
    model.W_2 = W_2;
    
    if exist('Errors', 'var')
        model.Errors = Errors;
    else
        model.Errors = [];
    end
    
    if exist('iterations', 'var')
        model.numIteration = iterations;
    else
        model.numIteration = 200;
    end

    if exist('learningRate', 'var')
        model.learningRate = learningRate;
    else
        model.learningRate = 0.001;
    end
    
    if exist('numHiddenUnits', 'var')
        model.numHiddenUnits = numHiddenUnits;
    else
        model.numHiddenUnits = 25;
    end    
end