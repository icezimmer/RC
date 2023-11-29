function [training_input_data, validation_input_data, training_target_data, validation_target_data] = splitDataset(input_data, target_data, fraction_to_validate)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Calculate fracion of the length of the cell array
validation_size = round(fraction_to_validate * length(target_data));

% Generate a random permutation of indices
randIndices = randperm(length(target_data));

% Select the first 20% of these indices for the 20% selection
validation_indices = randIndices(1:validation_size);

% Select the complementary indices for the 80% selection
training_indices = randIndices(validation_size + 1:end);

% Select the cells for both selections
validation_input_data = input_data(validation_indices);
training_input_data = input_data(training_indices);
validation_target_data = target_data(validation_indices);
training_target_data = target_data(training_indices);

end