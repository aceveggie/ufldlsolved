function opttheta = fineTuning(theta, inputSize, hiddenSize, numClasses, ...
                    netconfig, lambda, data, labels)
if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'MaxIter')
    options.MaxIter = 400;
end

[opttheta, cost] = minFunc(@(p) stackedAECost(p, inputSize, hiddenSize, ...
                   numClasses, netconfig, lambda, data, labels), theta, options);


myCostFunc = @(stackedAETheta) stackedAECost(theta(:), inputSize, hiddenSize, numClasses, netconfig, lambda, data, labels);
[stackedAETheta(:), cost] = fmincg(myCostFunc, stackedAETheta(:), options);

opttheta = stackedAETheta;

end