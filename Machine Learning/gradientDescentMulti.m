function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== CODE HERE ======================
    %
    tmp = theta;
    feature_dim = length(X(1,:));
    for i = 1: feature_dim
        k = 1;
        sum = 0;
      
        while( k <= m )
            sum = sum + ((theta)' * (X(k,:))' - y(k)) * X(k, i);
            k = k + 1;
        end
        tmp(i) = tmp(i) - alpha * sum / m;
    end
    
    theta = tmp;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    fprintf('---%d--- \r\n', J_history(iter));
    %make sure cost function J always goes down

end

end
