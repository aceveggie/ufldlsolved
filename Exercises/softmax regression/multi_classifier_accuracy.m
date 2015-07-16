function accuracy = multi_classifier_accuracy(theta, X,y)
	% theta' * X is the hypothesis
	x1 = X(:, 1);
	
	output = theta' * X;
	output = exp(bsxfun(@minus, output, max(output)));
	output = bsxfun(@rdivide, output, sum(output));
	[~, labels] = max(output);

	correct=sum(y == labels);
	accuracy = correct / length(y);
end