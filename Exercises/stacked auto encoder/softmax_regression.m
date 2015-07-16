function [cost, grad] = softmax_regression(theta, numClasses, inputSize, lambda, data, labels)

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;
thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
% Compute the prediction and the cost function
% Shift to prevent possible overflow
inner_prod = theta * data;
tmp_comp = exp(bsxfun(@minus, inner_prod, max(inner_prod)));
predict_prob = bsxfun(@rdivide, tmp_comp, sum(tmp_comp));
% Native cost function term
cost = sum(vec(-log(predict_prob) .* groundTruth)) / numCases;
% Regularization term
cost = cost + lambda/2 * sum(vec(theta.^2));

% Compute the gradient vector for softmax
for k = 1:numClasses
    thetagrad(k, :) = -((labels == k)' - predict_prob(k, :)) * data' / numCases + ...
                      lambda * theta(k, :);
end
% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end