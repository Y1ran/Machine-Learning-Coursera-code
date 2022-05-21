function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
sum = 0;

for i = 1:m
    cost = ((theta)' * (X(i,:))' - y(i)) ^ 2;
    sum = sum + cost;
end

J = 1/m * (1/2) * sum;
% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% sum = 0;
% temp_cost = X * theta - y;
% for i = 1:m
%     sum = sum + temp_cost(i)^2;
% end

% J = (1/(2*m))*sum;



% =========================================================================

end
