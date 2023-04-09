function W_hat = initStateMatrix(Nh, rho, seed, dns)
rng(seed)

W_hat = sprand(Nh,Nh, dns);
mask = W_hat~=0;
W_hat(mask) = 2*W_hat(mask) -1;

W_hat = rho * (W_hat / abs(eigs(W_hat, 1)));

end

