function [all_theta] = oneVsAll(X, y, num_labels, lambda)

%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

for i = 1 : num_labels
    y_tmp = (y == i);
    initial_theta = zeros(n + 1, 1);
    
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    %This function will return theta and the cost 
    [all_theta(i,:)] = ...
        fmincg (@(t)(lrCostFunction(t, X, y_tmp, lambda)), ...
                initial_theta, options);
end





% =========================================================================


end
