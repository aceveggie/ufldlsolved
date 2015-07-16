function [cost, grad] = orthonormalICACost(theta, visibleSize, numFeatures, patches, epsilon)
%orthonormalICACost - compute the cost and gradients for orthonormal ICA
%                     (i.e. compute the cost ||Wx||_1 and its gradient)
numPatches = size(patches, 2);
weightMatrix = reshape(theta, numFeatures, visibleSize);
mapping = weightMatrix * patches;

reconsError = norm(weightMatrix'*weightMatrix*patches - patches, 'fro')^2 / numPatches;
sparsityError = sum(sum(abs(mapping))) / numPatches;
cost = reconsError + sparsityError;

reconsGrad = 2/numPatches*weightMatrix*((patches*patches')*(weightMatrix'*weightMatrix) + ...
                             (weightMatrix'*weightMatrix)*(patches*patches') - ...
                             2*(patches*patches'));
sparsityGrad = sign(mapping) * patches' / numPatches; 
grad = reconsGrad + sparsityGrad;
grad = grad(:);
end