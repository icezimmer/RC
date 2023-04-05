classdef ReservoirComputing
    %UNTITLED Summary of this class goes here
    %   Detailed eobj.Hiddenplanation goes here

    properties
        InputDimension
        InputScaling
        NeuronsNumber
        InitialCondition
        SpectralRadius
        Density
        LeakyFactor
        WashOut
        Regularization
        Seed
        InputWeights
        HiddenWeights
        OutputWeights
    end

    methods
        function obj = ReservoirComputing(Nu, omega_in, Nh, x0, rho, dns, a, ws, lambda_r, seed)
            if a < 0 || a > 1
                error('The parameter a must be in [0, 1]')
            else
            obj.InputDimension = Nu;
            obj.InputScaling = omega_in;
            obj.NeuronsNumber = Nh;
            obj.InitialCondition = x0;
            obj.SpectralRadius = rho;
            obj.Density = dns;
            obj.LeakyFactor = a;
            obj.WashOut = ws;
            obj.Regularization = lambda_r;
            obj.Seed = seed;
            obj.InputWeights = initInputMatrix(Nu, omega_in, Nh, seed, a);
            obj.HiddenWeights = initStateMatrix(Nh, rho, seed, dns, a);
            obj.OutputWeights = [];
            end
        end


        function [hidden, hidden_washout] = hiddenState(obj, input_data)
            hidden = cell(size(input_data));
            hidden_washout = cell(size(input_data));

            num_samples = size(input_data,1);
            for index_sample=1:num_samples
                time_steps = size(input_data{index_sample},2);
                % Add ones to input for bias
                input_sample = zeros(obj.InputDimension+1,time_steps);
                input_sample(1:end-1,:) = input_data{index_sample};
                input_sample(end,:) = ones(1, time_steps);
               
                hidden_sample = zeros(obj.NeuronsNumber,time_steps);
                hidden_sample(:,1) = obj.InitialCondition;
                for t=1:time_steps
                    hidden_sample(:,t+1) = (1-obj.LeakyFactor)*hidden_sample(:,t) + obj.LeakyFactor*tanh(obj.InputWeights*input_sample(:,t) + obj.HiddenWeights*hidden_sample(:,t));
                end
                % Discard the initial state
                hidden_sample = hidden_sample(:, 2:end);
                hidden{index_sample} = hidden_sample;
                % Discard the washout
                hidden_sample = hidden_sample(:, obj.WashOut+1:end);
                hidden_washout{index_sample} = hidden_sample;
            end
        end

        function [obj, prediction, hidden, hidden_washout] = fit(obj, input_data, target_data, num_classes)
            [hidden, hidden_washout] = hiddenState(obj, input_data);
            hidden_washout_mat = cell2mat(hidden_washout');
            hidden_mat = cell2mat(hidden');
            one_hot = eye(num_classes);
            target_data_t = target_data';
            target_data_mat = one_hot(:,[target_data_t{:}]);

            obj.OutputWeights = trainOffline(hidden_washout_mat, target_data_mat, obj.Regularization, obj.WashOut);
            output_mat = readout(hidden_mat, obj.OutputWeights);
            [~, prediction_mat] = max(output_mat,[],1);

            num_samples = size(input_data,1);
            prediction = cell(num_samples,1);
            prec = 0;
            for index_sample=1:num_samples
                time_steps = size(input_data{index_sample},2);
                prediction_sample = prediction_mat(prec+1:prec+time_steps);
                prediction{index_sample} = categorical(prediction_sample);
                prec = prec + time_steps;
            end
        end


        function [prediction, hidden] = classify(obj, input_data)
            hidden = hiddenState(obj, input_data);
            hidden_mat = cell2mat(hidden');

            output_mat = readout(hidden_mat, obj.OutputWeights);
            [~, prediction_mat] = max(output_mat,[],1);

            num_samples = size(input_data,1);
            prediction = cell(num_samples,1);
            prec = 0;
            for index_sample=1:num_samples
                time_steps = size(input_data{index_sample},2);
                prediction_sample = prediction_mat(prec+1:prec+time_steps);
                prediction{index_sample} = categorical(prediction_sample);
                prec = prec + time_steps;
            end
        end

    end

end