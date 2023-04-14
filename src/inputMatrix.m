function W_in = inputMatrix(Nu, omega_in, Nh, seed)
rng(seed)

W_in = 2*rand(Nh,Nu) - 1;
W_in = omega_in * W_in;

end

