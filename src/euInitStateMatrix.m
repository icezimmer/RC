function W_hat = euInitStateMatrix(Nh, omega_r, seed)
rng(seed)

% EuSN
W = omega_r * (2*rand(Nh,Nh) - 1);
W_hat = W - W';

end