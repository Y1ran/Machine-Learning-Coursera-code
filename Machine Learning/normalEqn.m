function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================

%compra to Gradient Descent, here only 1 line code! what the f..
solution = pinv((X'* X)) * X'* y;

% -------------------------------------------------------------

theta = solution
% ============================================================

end
