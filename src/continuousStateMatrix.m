function W_hat = continuousStateMatrix(Nh, eigenvalues, seed)
rng(seed)

D = diag(eigenvalues);
V = rand(Nh); 
W_hat = V * (D / V); %W_hat = V*(D*inv(V))
%W_hat = V \ (D*V); %W_hat = inv(V)*(D*V)

end

