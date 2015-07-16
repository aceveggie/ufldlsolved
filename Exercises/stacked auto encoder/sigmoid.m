function val = sigmoid(z)
    val = 1 ./ (1 + exp(-z));
end