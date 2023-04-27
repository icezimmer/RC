function input_aug = inputAugmentation(input, Nh)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
input = input(:);
Nu = length(input);

if Nh > Nu 
    input_aug = cat(1,input,zeros(Nh-Nu,1));
elseif Nh == Nu
    input_aug = input;
else
    error('Dimension of hidden state must be greater or equal to dimension of input')
end