function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
% theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
% taking num_iters gradient steps with learning rate alpha
% Initialize some useful values

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== CODE HERE ======================
    %Notice it has been updated simultaneously otherwise the value
    %will has little disparity in theta around [0.006,0.0006]
    tmp = theta;
    for i = 1:2
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
    J_history(iter) = computeCost(X, y, theta);
    fprintf('---%d--- \r\n', J_history(iter));
    %make sure cost function J always goes down

end

end
