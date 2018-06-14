function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)


[row, col] = size(X);


cost = (-1 .* y)' * log(sigmoid( X * theta)) ...
       -(ones(row,1) - y)' * log(ones(row,1) - sigmoid(X * theta));
%penal =  ((norm(theta))^2);
%J = J + cost;

thet = theta(2:col,1);

penal =  lambda / 2 * ((norm(thet))^2);

J = 1 / m * (cost + penal);



tmp = X' * (sigmoid(X* theta) - y);
%grad(j) = 1 / m * tmp;
%sums = zeros(col,1);

theta(1,1) = 0; 
pena = lambda .* theta;
grad = tmp + pena;



grad = (1 / m) .* grad(:);
    
end

