addpath src

[input_data, target_data] = dataLoader();
[~, time_steps, num_samples] = size(input_data);

% Shuffle the dataset
seed_shuffle = 13;
rng(seed_shuffle)
shuffle = randperm(num_samples);
input_data = input_data(:,:,shuffle);
target_data = target_data(:,shuffle);

dv_index = 1:64; % from 1 to 64
ts_index = 65:88; % from 65 to 88

dv_in = input_data(:,:,dv_index);
ts_in = input_data(:,:,ts_index);

dv_tg = target_data(:,dv_index);
dv_tg = kron(dv_tg, ones(1,time_steps));
ts_tg = target_data(:,ts_index);
ts_tg = kron(ts_tg, ones(1,time_steps));

[dv_layer_in, dv_layer_tg] = dataProcess(dv_in, dv_tg);
[ts_layer_in, ts_layer_tg] = dataProcess(ts_in, ts_tg);

Nu = size(input_data, 1);
omega_in = 0.4; 
omega_b = 0.2;
Nh = 400;
f = @(bias, input_weights, input, hidden_weights, hidden) tanh(bias + input_weights*input + hidden_weights*hidden);
x0 = zeros(Nh,1);
phi = @rungeKutta;
eps = 0.0001;
rho = 1.1;
dns = 0.1;
%a = 0.1;
ws = 0;
lambda_r = 0.1; 
%Nl = 1;
seed = 1;

rc=ReservoirComputing(Nu, omega_in, omega_b, Nh, f, x0, phi, eps, rho, dns, ws, lambda_r, seed);

% hid = rc.hiddenState(dv_layer_in);
% 
% layers = [ ...
%     sequenceInputLayer(Nh)
%     gruLayer(10, OutputMode="sequence")
%     fullyConnectedLayer(7)
%     softmaxLayer
%     classificationLayer];
% options = trainingOptions('adam', ...
%     MiniBatchSize=size(dv_layer_tg, 1), ...
%     MaxEpochs=100, ...
%     GradientThreshold=2, ...
%     shuffle='never', ...
%     Verbose=1);
% [net, info] = trainNetwork(hid,dv_layer_tg,layers,options);

[rc,pred_tr]=rc.fit(dv_layer_in,dv_layer_tg,7);
figure
confusionchart([dv_layer_tg{:,:}], [pred_tr{:,:}]);
title("TR")

pred_ts=rc.classify(ts_layer_in);
figure
confusionchart([ts_layer_tg{:,:}], [pred_ts{:,:}]);
title("TS")

[loss, accuracy_K_ts, accuracy_ts, accuracy_av_ts, F1_ts, F1_macro_ts] = evaluation(ts_layer_tg, pred_ts);
