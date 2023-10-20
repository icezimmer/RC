function input_aug = inputAugmentation(input_data, Nz)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Nx = size(input_data,1);

if Nz > Nx 
    input_aug = cat(1,input_data,zeros(Nz-Nx,size(input_data,2)));
elseif Nz == Nx
    input_aug = input_data;
else
    error('Dimension of hidden state must be greater or equal to dimension of input')
end