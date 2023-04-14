function W_hat = continuousStateMatrix(Nh, eigenvalues, seed)
rng(seed)

D = diag(eigenvalues);
V = rand(Nh);

W_hat = V \ (D*V);

end

