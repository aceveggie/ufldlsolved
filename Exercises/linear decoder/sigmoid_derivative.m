function val = sigmoid_derivative(z)
	val = sigmoid(z) .* (1 - sigmoid(z));
end
