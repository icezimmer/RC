addpath src

[input_data, target_data] = dataLoader();
[~, ~, num_samples] = size(input_data);

% Shuffle the dataset
seed_shuffle = 13;
rng(seed_shuffle)
shuffle = randperm(num_samples);
input_data = input_data(:,:,shuffle);
target_data = target_data(:,shuffle);

%omega_in, Nh, x0, rho, dns, a, ws, lambda_r, seed
rc=ReservoirComputing(0.4, 500, zeros(500,1), 0.9, 0.1, 0.1, 0, 0.1, 1);
rc=rc.fit(input_data(:,:,1:64),target_data(:,1:64));
rc=rc.predict(input_data(:,:,65:88));
rc.confusion_matrix(target_data(:,65:88),size(input_data,2));