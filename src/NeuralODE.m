classdef NeuralODE
    %UNTITLED Summary of this class goes here
    %   Detailed eobj.Hiddenplanation goes here

    properties
        InputNetwork
        HiddenSize
        OutputNetwork
        BiasScaling
        TimeSteps
        OdeFunction
        NumericalMethod
        StepSize
        Spectrum
        Transient
        Regularization
        %LayersNumber
        Seed
        Bias
        HiddenWeights
        %HiddenHiddenWeights
        OutputWeights
    end

    methods
        function obj = NeuralODE(f_in, Nz, f_out, omega_b, T, f, phi, eps, eigs, ws, lambda_r, seed)
            obj.InputNetwork = f_in;
            obj.HiddenSize = Nz;
            obj.OutputNetwork = f_out;
            obj.BiasScaling = omega_b;
            obj.TimeSteps = floor(T/eps);
            obj.OdeFunction = f;
            obj.NumericalMethod = phi;
            obj.StepSize = eps;
            obj.Spectrum = eigs;
            obj.Transient = ws;
            obj.Regularization = lambda_r;
            %obj.LayersNumber = Nl;
            obj.Seed = seed;
            obj.Bias = [];
            obj.HiddenWeights = continuousStateMatrix(eigs, seed);
            %obj.HiddenHiddenWeights = initInputMatrix(Nh, 1, Nh, seed, a);
            obj.OutputWeights = [];
        end


        function [hidden, hidden_washout, pooler] = hiddenState(obj, input_data)
            obj.Bias = bias(length(obj.HiddenWeights), obj.BiasScaling, obj.Seed);
            input_data_mat = cell2mat(input_data');

            hidden_mat = zeros([length(obj.HiddenWeights), length(input_data), obj.TimeSteps]);
            hidden_mat_0 = obj.InputNetwork(input_data_mat);

            hidden_mat(:,:,1) = hidden_mat_0;
            for t=1:obj.TimeSteps
                %hidden_mat(:,:,t+1) = obj.NumericalMethod(obj.Bias, obj.HiddenWeights, hidden_mat(:,:,t), obj.OdeFunction, obj.StepSize);
                hidden_mat(:,:,t+1) = hidden_mat(:,:,t) + obj.StepSize * obj.OdeFunction(obj.Bias, obj.HiddenWeights, hidden_mat(:,:,t));
            end

            hidden_washout_mat = hidden_mat(:,:,obj.Transient+1:end);
            pooler_mat = hidden_mat(:,:,end);
            % Convert to cell array
            hidden = arrayfun(@(k) squeeze(hidden_mat(:,k,:)), 1:size(hidden_mat,2), 'UniformOutput', false);
            hidden_washout = arrayfun(@(k) squeeze(hidden_washout_mat(:,k,:)), 1:size(hidden_washout_mat,2), 'UniformOutput', false);
            pooler = arrayfun(@(k) squeeze(pooler_mat(:,k,:)), 1:size(pooler_mat,2), 'UniformOutput', false);
            % Transpose the cell array to get N x 1 shape
            hidden = hidden';
            hidden_washout = hidden_washout';
            pooler = pooler';


            % hidden = cell(size(input_data,1), 1);
            % hidden_washout = cell(size(input_data,1), 1);
            % pooler = cell(size(input_data,1), 1);
            % % hidden = cell(size(input_data,1), obj.LayersNumber);
            % % hidden_washout = cell(size(input_data,1), obj.LayersNumber);
            % 
            % num_samples = size(input_data,1);
            % for index_sample=1:num_samples
            %     input_sample = input_data{index_sample};
            % 
            %     % if size(input_sample, 2) > 1
            %     %     input_sample = input_sample(:);
            %     % end
            %     input_sample = double(input_sample);
            %     hidden_sample_0 = obj.InputNetwork(input_sample);
            % 
            %     obj.Bias = bias(length(hidden_sample_0), obj.BiasScaling, obj.Seed);
            %     hidden_sample = zeros(length(hidden_sample_0), 1+obj.TimeSteps);
            % 
            %     hidden_sample(:,1) = hidden_sample_0;
            %     for t=1:obj.TimeSteps
            %         hidden_sample(:,t+1) = obj.NumericalMethod(obj.Bias, obj.HiddenWeights, hidden_sample(:,t), obj.OdeFunction, obj.StepSize);
            %     end
            % 
            %     pooler{index_sample, 1} = hidden_sample(:,end);
            %     hidden{index_sample,1} = hidden_sample;
            %     % Discard the transient
            %     hidden_sample = hidden_sample(:, obj.Transient+1:end);
            %     hidden_washout{index_sample,1} = hidden_sample;
            % end
        end


        function obj = fitInitialCondition(obj, input_data, target_data)
            [~, ~, pooler] = hiddenState(obj, input_data);
            pooler_mat = cell2mat(pooler');
            target_data_mat = onehotencode(target_data', 1);


            obj.OutputWeights = trainOffline(pooler_mat, target_data_mat, obj.Regularization);

            % Define the adjoint ODE: da(t)/dt = -a(t)'*df/dh
            adjoint_ODE = @(x) -x'*obj.HiddenWeights;

            % Compute the initial condition a(T):
            % a(T) = dL/dh(T) = ((y - d)' * OutputWeights(:,1:end-1))' =
            % = (OutputWeights(:,1:end-1))' * (y - d)
            % We leave the last column to leave the bias weights
            adjoint_mat_T = (obj.OutputWeights(:,1:end-1))' * (obj.OutputNetwork(pooler_mat, obj.OutputWeights) - target_data_mat);

            % Discretization of the solution a(t): from a(T) to a(0)
            adjoint_mat = zeros([length(obj.HiddenWeights), length(target_data), obj.TimeSteps]);
            adjoint_mat(:,:,1) = adjoint_mat_T;
            for t=1:obj.TimeSteps
                adjoint_mat(:,:,t+1) = adjoint_mat(:,:,t) + (obj.StepSize * adjoint_ODE(adjoint_mat(:,:,t)))';
            end

            % Upgrade the initial condition h(0) by SGD:
            % h(0)_new = h(0)_old - dL/dh(0) = h(0)_old - a(0)
            obj.InputNetwork = @(x) obj.InputNetwork(x) - adjoint_mat(:,:,end);
        end


        function prediction = classify(obj, input_data)
            [~, ~, pooler] = hiddenState(obj, input_data);
            pooler_mat = cell2mat(pooler');

            output_mat = readout(pooler_mat, obj.OutputWeights);
            [~, prediction_mat] = max(output_mat,[],1);

            num_samples = size(input_data,1);
            prediction = cell(num_samples,1);
            prec = 0;
            for index_sample=1:num_samples
                % time_steps is equal to 1
                prediction_sample = prediction_mat(prec+1);
                prediction{index_sample} = categorical(prediction_sample);
                prec = prec + 1;
            end
        end


    end

end