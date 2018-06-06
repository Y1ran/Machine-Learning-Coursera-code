function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

[row, col] = size(X);

for i = 1:row
    cost = (-y(i)) * log(sigmoid(theta' * X(i,:)')) ...
        -(1 - y(i)) * log(1 - sigmoid(theta' * X(i,:)'));
    J = J + cost;
end

J = 1 / m * J;


% for j = 1: length(theta)
%     tmp = (sigmoid(X(:,j) * theta(j,:))' - y') * X(:,j);
%     grad(j) = 1 / m * tmp;
% end

sum = zeros(col,1);

for j = 1: col
    for i = 1: m
        tmp = (sigmoid(X(i,:) * theta) - y(i)) * X(i,j);
        sum(j) = sum(j) +  tmp;

grad = (1 / m) * sum;
    end
% =============================================================

end
