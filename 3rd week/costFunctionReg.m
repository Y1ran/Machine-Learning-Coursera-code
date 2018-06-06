function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
[row, col] = size(X);

for i = 1:row
    cost = (-y(i)) * log(sigmoid(theta' * X(i,:)')) ...
        -(1 - y(i)) * log(1 - sigmoid(theta' * X(i,:)'));
    %penal =  ((norm(theta))^2);
    J = J + cost;
end
thet = theta(2:col,1);

penal =  lambda / 2 * (norm(thet))^2;

J = 1 / m * (J + penal);


% for j = 1: length(theta)
%     tmp = (sigmoid(X(:,j) * theta(j,:))' - y') * X(:,j);
%     grad(j) = 1 / m * tmp;
% end

sums = zeros(col,1);

for j = 1: col
    if(j == 1)
        for i = 1: m
            tmp = (sigmoid(X(i,:) * theta) - y(i)) * X(i,j);
            sums(j) = sums(j) +  tmp;
        end
    else
        for i = 1: m
            tmp = (sigmoid(X(i,:) * theta) - y(i)) * X(i,j);
            
            sums(j) = sums(j) +  tmp;
        end
        penal = lambda * theta(j);
        sums(j) = sums(j) + penal;
    
    end

grad = (1 / m) * sums;
    
end


end
