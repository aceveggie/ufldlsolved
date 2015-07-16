function [softMaxOutput] = softmaxTrain(inputData, inputLabels, layerSize, numClasses, sLambda, options);

	m = size(inputData, 2);
	n = size(inputData, 1);
	theta = 0.005 * randn( numClasses* layerSize, 1);
	inputLabels = inputLabels';
		
	tic;

	myCostFunc = @(theta) softmax_regression(theta(:), ...
	 										numClasses, ...
	 										layerSize, sLambda, inputData, inputLabels);
	[theta(:), cost] = fmincg(myCostFunc, theta(:), options);
	fprintf('Optimization took %f seconds.\n', toc);
	optimizedTheta = reshape(theta, numClasses, layerSize);


	fprintf('Optimization took %f seconds.\n', toc);
	toc;

	softMaxOutput.layerSize = layerSize;
	softMaxOutput.optimizedTheta = optimizedTheta;
	softMaxOutput.numClasses = numClasses;
end