function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  theta = [theta, zeros(size(theta,1),1)]; % n x num_classes(785 * 10)
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%

% M = exp(theta' * X);
% Msum = sum(M, 1);
% Py = M ./Msum;
% logPy = log(Py);
% size(theta)
% size(X)
% size(M)
% size(Msum)
% size(theta' * X)
% size(Py)
% MM = M(:, 1:20);
% MM = MM ./sum(MM, 1);
% MM
% Py(:, 1:20)
% unique(y)
% JSum = 0
% for i = 1:num_classes
%   z = (y == i);
%   size(z * log(Py'))
%   sum(z* log(Py'))
% end

  h = exp(theta' * X);  % num_classes x m (10 * 60000)
  % we are using sub2ind because, we need sum up all the cost
  I = sub2ind(size(h), y, 1:size(h,2));
  % normalize
  f = -sum(log(h(I) ./ sum(h,1)));
  % f contains the cost

  p = bsxfun(@rdivide, h, sum(h,1)); % k x m(10 * 60000)
  % p = h ./ sum(h, 1);
  val_k = full(sparse(y, 1:m, 1)); % k x m
  g = g - X * (val_k - p)'; % n x n
  g(:,end) = [];

  g=g(:); % make gradient a vector for minFunc

end

% gradient is instatneous slope of the error function at the specified weight