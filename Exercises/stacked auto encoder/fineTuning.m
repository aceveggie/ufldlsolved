function opttheta = fineTuning(theta, inputSize, hiddenSize, numClasses, ...
                    netconfig, lambda, data, labels)
if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'MaxIter')
    options.MaxIter = 400;
end

myCostFunc = @(theta) stackedAECost(theta(:), inputSize, hiddenSize, numClasses, netconfig, lambda, data, labels);
[theta(:), cost] = fmincg(myCostFunc, theta(:), options);

opttheta = theta;

end