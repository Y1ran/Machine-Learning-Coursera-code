function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.


% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


Ck = zeros(K,1);

for j = 1:K
    for i = 1:m
        if(idx(i) == j)
            Ck(j) = Ck(j) + 1;
        end
    end
end


for j = 1:K
    Mu = zeros(K, n);
    for i = 1:m
        
        if (idx(i) == j)
            Mu(i,:) = X(i,:);
        end
    end
        
    for k = 1:n
        centroids(j,k) = 1 / Ck(j) * sum(Mu(:,k));
            
            
    end
    
end






% =============================================================


end

