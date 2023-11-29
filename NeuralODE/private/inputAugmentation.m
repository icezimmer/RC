function input_aug = inputAugmentation(input_data, hidden_dim)
% WARNING!! the inputnetwork depens on the num_samples. This is a
% problem!!!
[input_dim, num_samples] = size(input_data); 

if hidden_dim > input_dim 
    input_aug = cat(1,input_data,zeros(hidden_dim-input_dim, num_samples));
elseif hidden_dim == input_dim
    input_aug = input_data;
else
    error('Dimension of hidden state must be greater or equal to dimension of input')
end