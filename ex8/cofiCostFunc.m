function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
cost = 0;
tmp_Y = zeros(num_movies, num_users);

Reg_Theta = 0;
Reg_X = 0;

for i = 1:num_movies
    for k = 1:num_features
        
    %X_tmp = X(idx,:);
        Reg_X =  Reg_X + norm(X(i,k)) ^ 2;
        
    end
end

for j = 1:num_users
    for k = 1:num_features    
        
    %X_tmp = X(idx,:);
        Reg_Theta =  Reg_Theta + norm(Theta(j,k)) ^ 2;
        
    end
end



for i = 1:num_movies
    for j = 1:num_users
        if(R(i,j) == 1)
            cost = cost + (Theta(j,:)*X(i,:)' - Y(i,j)) ^ 2;
        end
    end
end

J = 1 / 2 * (cost + lambda * (Reg_Theta + Reg_X));


% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
grad_X = zeros(size(X));
grad_Theta = zeros(size(Theta));

for i = 1:num_movies
    
    idx = find(R(i,:) == 1);
    Theta_tmp = Theta(idx,:);
    Y_tmp = Y(i,idx);
    
    grad_X(i,:) = (X(i,:) * Theta_tmp' - Y_tmp) * Theta_tmp + ...
        lambda * X(i,:);
    for j = 1:num_users
        jdx = find(R(:,j) == 1);
        X_tmp = X(jdx,:);
        Y_tmp = Y(jdx,j);
        grad_Theta(j,:) = (Theta(j,:) * X_tmp' - Y_tmp') * X_tmp + ...
            lambda * Theta(j,:);
    end
end




% =============================================================

grad = [grad_X(:); grad_Theta(:)];

end
