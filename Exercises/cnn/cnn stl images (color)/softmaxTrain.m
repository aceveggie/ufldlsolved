function [softmaxModel] = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options)
%softmaxTrain Train a softmax model with the given parameters on the given
% data. Returns softmaxOptTheta, a vector containing the trained parameters
% for the model.
%
% inputSize: the size of an input vector x^(i)
% numClasses: the number of classes 
% lambda: weight decay parameter
% inputData: an N by M matrix containing the input data, such that
%            inputData(:, c) is the cth input
% labels: M by 1 matrix containing the class labels for the
%            corresponding inputs. labels(c) is the class label for
%            the cth input
% options (optional): options
%   options.maxIter: number of iterations to train for


if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'MaxIter')
    options.MaxIter = 400;
end

% initialize parameters
theta = 0.005 * randn(numClasses * inputSize, 1);

% Use minFunc to minimize the function
disp('optimizing the cost function for softmax regression');

% size(theta)
% size(inputSize)
% size(inputData)
% size(labels)
% disp('-------')

myCostFunc = @(theta) softmax_regression(theta(:), numClasses, inputSize, lambda, inputData, labels);
[theta(:), cost] = fmincg(myCostFunc, theta(:), options);

softmaxOptTheta = theta;
% Fold softmaxOptTheta into a nicer format
softmaxModel.optimizedTheta = reshape(softmaxOptTheta, numClasses, inputSize);
softmaxModel.inputSize = inputSize;
softmaxModel.numClasses = numClasses;
                          
end                          