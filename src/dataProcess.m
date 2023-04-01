function [input,target] = dataProcess(input,target)
%DATAPROCESS Summary of this function goes here
%   Detailed explanation goes here
input=num2cell(input,[1,2]);
input=reshape(input,[size(input,1) * size(input,3), 1]);

[target, ~] = find(target);
target=reshape(target,[480, size(target,1)/480])';
target = categorical(target);
target=num2cell(target,2);
end

