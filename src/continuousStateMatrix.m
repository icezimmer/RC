function W_hat = continuousStateMatrix(Nh, eigenvalues, seed)
rng(seed)

%D = diag(eigenvalues);
r = 1;
theta = pi/2;
block = r*[cos(theta), sin(theta); -sin(theta), cos(theta)];
cell_block = repmat({block}, 1, Nh/2);
D = blkdiag(cell_block{:});
V = rand(Nh); 
W_hat = V * (D / V);
W_hat=D;

%W_hat = V - V';

end

