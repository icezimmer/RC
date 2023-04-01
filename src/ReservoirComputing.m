classdef ReservoirComputing
    %UNTITLED Summary of this class goes here
    %   Detailed eobj.Hiddenplanation goes here

    properties
        InputScaling
        NeuronsNumber
        InitialCondition
        SpectralRadius
        Density
        LeakyFactor
        WashOut
        Regularization
        Seed
        Hidden
        Pooler
        Output
        InputWeights
        HiddenWeights
        OutputWeights
    end

    methods
        function obj = ReservoirComputing(omega_in, Nh, x0, rho, dns, a, ws, lambda_r, seed)
            if a < 0 || a > 1
                error('The parameter a must be in [0, 1]')
            else
            obj.InputScaling = omega_in;
            obj.NeuronsNumber = Nh;
            obj.InitialCondition = x0;
            obj.SpectralRadius = rho;
            obj.Density = dns;
            obj.LeakyFactor = a;
            obj.WashOut = ws;
            obj.Regularization = lambda_r;
            obj.Seed = seed;
            obj.Hidden = [];
            obj.Pooler = [];
            obj.InputWeights = [];
            obj.HiddenWeights = initStateMatrix(Nh, rho, seed, dns, a);
            obj.OutputWeights = [];
            end
        end


        function obj = fit(obj, input_data, target_data)          
            [dim_input, time_steps, num_samples] = size(input_data);
            [num_classes, ~] = size(target_data);
            one_hot = eye(num_classes);

            obj.InputWeights = initInputMatrix(dim_input, obj.InputScaling, obj.NeuronsNumber, obj.Seed, obj.LeakyFactor);

            % For each sample repeat the category sample at each timestep
            target_data = kron(target_data, ones(1,time_steps));
                
            obj.Hidden = [];
            for sample=1:num_samples
                hidden_sample = obj.InitialCondition;
                % Add ones to input for bias
                input_sample = [input_data(:,:,sample); ones(1, time_steps)];
                for t=1:time_steps
                    hidden_sample = cat(2, hidden_sample, (1-obj.LeakyFactor)*hidden_sample(:,end) + obj.LeakyFactor*tanh(obj.InputWeights*input_sample(:,t) + obj.HiddenWeights*hidden_sample(:,end)));
                end
                % Discard the initial state
                hidden_sample = hidden_sample(:, 2:end);
                obj.Hidden = cat(2,obj.Hidden,hidden_sample);
            end

            obj.OutputWeights = trainOffline(obj.Hidden, target_data, obj.Regularization, obj.WashOut);
            y_tr = readout(obj.Hidden, obj.OutputWeights);
            [~, argmax_tr] = max(y_tr,[],1);
            obj.Output = one_hot(:,argmax_tr);
        end


        function obj = predict(obj, input_data)
            [~, time_steps, num_samples] = size(input_data);
            num_classes = size(obj.OutputWeights, 1);
            one_hot = eye(num_classes);

            obj.Hidden = [];
            for sample=1:num_samples
                hidden_sample = obj.InitialCondition;
                % Add ones to input for bias
                input_sample = [input_data(:,:,sample); ones(1, time_steps)];
                for t=1:time_steps
                    hidden_sample = cat(2, hidden_sample, (1-obj.LeakyFactor)*hidden_sample(:,end) + obj.LeakyFactor*tanh(obj.InputWeights*input_sample(:,t) + obj.HiddenWeights*hidden_sample(:,end)));
                end
                % Discard the initial state
                hidden_sample = hidden_sample(:, 2:end);
                obj.Hidden = cat(2,obj.Hidden,hidden_sample);
            end

            y_ts = readout(obj.Hidden, obj.OutputWeights);
            [~, argmax_ts] = max(y_ts,[],1);
            obj.Output = one_hot(:,argmax_ts);
        end

        function confusion_matrix(obj, target_data, time_steps)
            % For each sample repeat the category sample at each timestep
            size(target_data)
            target_data = kron(target_data, ones(1,time_steps));
            size(target_data)
            % Plot Confusion Matrix
            [classes_target, ~] = find(target_data);
            [classes_predict, ~] = find(obj.Output);
            confusionchart(classes_target, classes_predict);
            title("Confusion Matrix (TS set)")
        end
    end

end