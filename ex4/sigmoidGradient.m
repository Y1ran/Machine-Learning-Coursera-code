function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

[rows cols] = size(g);

if cols == 1 && rows == 1
    g = sigmoid(z) * (ones(size(g)) - sigmoid(z));
else
    g = sigmoid(z) .* (ones(size(g)) - sigmoid(z));



% =============================================================




end
