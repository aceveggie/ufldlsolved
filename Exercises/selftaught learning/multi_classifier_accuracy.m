function accuracy = multi_classifier_accuracy(theta, X,y)
  % theta' * X is the hypothesis
  [~,labels] = max(theta'*X, [], 1);
  correct=sum(y == labels);
  accuracy = correct / length(y);
end