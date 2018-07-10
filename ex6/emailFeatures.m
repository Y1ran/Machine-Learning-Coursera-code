function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

% You need to return the following variables correctly.
x = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
len = length(word_indices);

for i = 1 : n
    for j = 1 : len
        if( i ==  word_indices(j,:))
            x(i,1) = 1;
        end
    end
end

% % This is the second method to compute
% for i = 1:len
%     tmp = word_indices(i,:);
%     for j = 1:n
%         if(tmp == j)
%             x(j,1) = 1;
%         end
%     end
% % end




% =========================================================================
    

end
