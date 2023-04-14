classdef ReservoirComputing
    %UNTITLED Summary of this class goes here
    %   Detailed eobj.Hiddenplanation goes here

    properties
        InputDimension
        InputScaling
        BiasScaling
        NeuronsNumber
        OdeFunction
        InitialCondition
        NumericalMethod
        StepSize
        SpectralRadius
        Density
        %LeakyFactor
        WashOut
        Regularization
        %LayersNumber
        Seed
        Bias
        InputWeights
        HiddenWeights
        %HiddenHiddenWeights
        OutputWeights
    end

    methods
        function obj = ReservoirComputing(Nu, omega_in, omega_b, Nh, f, x0, phi, eps, rho, dns, ws, lambda_r, seed)
            %if a < 0 || a > 1
            %    error('The parameter a must be in [0, 1]')
            %else
            obj.InputDimension = Nu;
            obj.InputScaling = omega_in;
            obj.BiasScaling = omega_b;
            obj.NeuronsNumber = Nh;
            obj.OdeFunction = f;
            obj.InitialCondition = x0;
            obj.NumericalMethod = phi;
            obj.StepSize = eps;
            obj.SpectralRadius = rho;
            obj.Density = dns;
            %obj.LeakyFactor = a;
            obj.WashOut = ws;
            obj.Regularization = lambda_r;
            %obj.LayersNumber = Nl;
            obj.Seed = seed;
            obj.Bias = bias(Nh, omega_b, seed);
            obj.InputWeights = inputMatrix(Nu, omega_in, Nh, seed);
            %obj.HiddenWeights = discreteStateMatrix(Nh, rho, seed, dns);
            obj.HiddenWeights = continuousStateMatrix(Nh, -Nh:-1, seed);
            %obj.HiddenHiddenWeights = initInputMatrix(Nh, 1, Nh, seed, a);
            obj.OutputWeights = [];
            %end
        end


        function [hidden, hidden_washout] = hiddenState(obj, input_data)
            hidden = cell(size(input_data,1), 1);
            hidden_washout = cell(size(input_data,1), 1);
            %hidden = cell(size(input_data,1), obj.LayersNumber);
            %hidden_washout = cell(size(input_data,1), obj.LayersNumber);

            num_samples = size(input_data,1);
            for index_sample=1:num_samples
                time_steps = size(input_data{index_sample},2);
                input_sample = input_data{index_sample};
               
                hidden_sample = zeros(obj.NeuronsNumber,time_steps);
                hidden_sample(:,1) = obj.InitialCondition;
                for t=1:time_steps
                    %hidden_sample(:,t+1) = (1-obj.LeakyFactor)*hidden_sample(:,t) + obj.LeakyFactor*tanh(obj.InputWeights*input_sample(:,t) + obj.HiddenWeights*hidden_sample(:,t));
                    hidden_sample(:,t+1) = obj.NumericalMethod(obj.Bias, obj.InputWeights, input_sample(:,t), obj.HiddenWeights, hidden_sample(:,t), obj.OdeFunction, obj.StepSize);
                end
                % Discard the initial state
                hidden_sample = hidden_sample(:, 2:end);
                hidden{index_sample,1} = hidden_sample;
                % Discard the washout
                hidden_sample = hidden_sample(:, obj.WashOut+1:end);
                hidden_washout{index_sample,1} = hidden_sample;
            end

%             for layer=1:obj.LayersNumber-1
%                 for index_sample=1:num_samples
%                     time_steps = size(input_data{index_sample},2);
%                     % Add ones to input for bias
%                     input_sample = zeros(obj.NeuronsNumber+1,time_steps);
%                     input_sample(1:end-1,:) = hidden{index_sample,layer};
%                     input_sample(end,:) = ones(1, time_steps);
%                    
%                     hidden_sample = zeros(obj.NeuronsNumber,time_steps);
%                     hidden_sample(:,1) = obj.InitialCondition;
%                     for t=1:time_steps
%                         hidden_sample(:,t+1) = (1-obj.LeakyFactor)*hidden_sample(:,t) + obj.LeakyFactor*tanh(obj.HiddenHiddenWeights*input_sample(:,t) + obj.HiddenWeights*hidden_sample(:,t));
%                     end
%                     % Discard the initial state
%                     hidden_sample = hidden_sample(:, 2:end);
%                     hidden{index_sample,layer+1} = hidden_sample;
%                     % Discard the washout
%                     hidden_sample = hidden_sample(:, obj.WashOut+1:end);
%                     hidden_washout{index_sample,layer+1} = hidden_sample;
%                 end
%             end
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