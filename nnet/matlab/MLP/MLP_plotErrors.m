function MLP_plotErrors(model)
    % plot error improvement during the training
    plot((1:model.numIteration), model.Errors);
    title(sprintf('Hidden Unit=%d, Learning rate=%f', model.numHiddenUnits, model.learningRate));
    xlabel('Iterations');
    ylabel('Squared Loss Error');
end