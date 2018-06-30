function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 

% Number of training examples
m = size(X, 1);
%X = [ones(m, 1) X];
% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);


%[theta] = trainLinearReg(X, y, lambda);

% Compute train/cross validation errors 
% for i = 1 : m
%     [theta] = trainLinearReg(X(1:i,:), y(1:i), lambda);
%     error_train(i) = linearRegCostFunction(X...
%         (1:i,:), y(1:i), theta, lambda);
%     error_val(i) = linearRegCostFunction(Xval, yval, theta, lambda);
% end
for i = 1:m
    X_sub = X(1:i, :);
    y_sub = y(1:i); 

    theta = trainLinearReg(X_sub, y_sub, lambda);

    error_train(i) = linearRegCostFunction(X_sub, y_sub, theta, 0);
    error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end
% ====================== YOUR CODE HERE ======================

% -------------------------------------------------------------

% =========================================================================

end
