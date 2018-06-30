function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%X = [ones(m,1) X];
% ====================== YOUR CODE HERE ======================
% cost = 0;
% total = 0;
% 
% for i =1:m
%     cost = (theta' * X(i,:)' - y(i,:))^2;
%     total = total + cost;
    

% cost = (norm((X * theta - y))) .^2;
% penal = lambda ./ (norm(theta(2:end,1))) .^ 2;
% 

J = (1/(2*m))*sum(power((X*theta - y),2))+ (lambda/(2*m)) * sum(power(theta(2:end),2));

G = (lambda/m) .* theta;
G(1) = 0; % this is always 0

grad = ((1/m) .* X' * (X*theta - y)) + G;
%J = 0.5 / m * (cost + penal);
% J = (1/(2*m))*sum(power((X*theta - y),2))+ (lambda/(2*m)) * sum(power(theta(2:end),2));
% % h=X*theta;
% % thetas=theta(2:end,1);
% % J=1/(2*m).*sum((h-y).^2)+(lambda/(2.*m)).*sum(thetas.^2);
% 
% grad = 1 / m .* ((theta' * X' - y') * X + lambda .* theta')';
% grad(1,:) = 1 / m * (theta' * X' - y') * X(:,1);


% =========================================================================

grad = grad(:);

end
