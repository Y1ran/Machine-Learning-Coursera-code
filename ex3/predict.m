function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

X = [ones(m,1) X];
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
%hidden_act = zeros(size(X, 1), size(Theta1, 2));
hidden_layer = [ones(size(X, 1),1) sigmoid(X * Theta1')];

%hidden_layer = zeros(size(X, 1), size(hidden_act, 2));
%hidden_layer = sigmoid(hidden_act);

output_act = hidden_layer * Theta2';
output_layer = sigmoid(output_act);


p = max(output_layer, [], 2);

for i = 1 : m
    for j = 1 : num_labels
        if( output_layer(i, j) == p(i, :))
            if( j ~= 10)
                p(i,:) = j;
            else
                p(i,:) = 0;
            end
        end
    end
end  


% =========================================================================


end
