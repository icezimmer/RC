function b = initBias(Nh, omega_b, seed)
rng(seed)
b = omega_b * (2*rand(Nh,1)-1);
end

