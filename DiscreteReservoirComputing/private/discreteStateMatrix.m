function W_hat = discreteStateMatrix(Nh, rho, seed, dns, a)
rng(seed)

if a > 0 && a <= 1
    W_hat = sprand(Nh,Nh, dns);
    mask = W_hat~=0;
    W_hat(mask) = 2*W_hat(mask) -1;
    W_tilde = (1-a)*speye(Nh) + a*W_hat;
    % Force the dinamycs to be stable (contractive map)
    W_tilde = rho * (W_tilde / abs(eigs(W_tilde, 1)));
    W_hat = (1/a)*(W_tilde - (1-a)*speye(Nh));
else
    W_hat = sparse(Nh,Nh);
end

end

