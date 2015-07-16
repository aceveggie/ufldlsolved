function [pred] = softmaxPredict(softMaxOutput, data)

	layerSize = softMaxOutput.layerSize;
	theta = softMaxOutput.optimizedTheta;
	numClasses = softMaxOutput.numClasses;

	pred = zeros(1, size(data, 2));

	inner_prod = theta * data;
	inner_prod = bsxfun(@minus, inner_prod, max(inner_prod));
	pred_prob = exp(inner_prod);
	[~, pred] = max(pred_prob);
end