function W_hat = initStateMatrix(Nh, rho, seed, dns, a)
rng(seed)

%LI-ESN
if a > 0
    W_hat = 2*sprand(Nh,Nh, dns) - 1;
    W_tilde = (1-a)*speye(Nh) + a*W_hat;
    W_tilde = rho * (W_tilde / max(abs(eig(W_tilde))));
    W_hat = (1/a)*(W_tilde - (1-a)*speye(Nh));
else
    W_hat = sparse(Nh,Nh);
end

end

