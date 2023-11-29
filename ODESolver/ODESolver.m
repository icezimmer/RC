classdef ODESolver
    %ODESOLVER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        ODE
        DeltaTime
    end
    
    methods
        function obj = ODESolver(f, eps)
            %ODESOLVER Construct an instance of this class
            %   Detailed explanation goes here
            obj.ODE = f;
            obj.DeltaTime = eps;
        end
        
        function new_hidden = eulerForward(obj, bias, hidden_weights, hidden, varargin)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            numvarargs = length(varargin);
            if numvarargs > 2
                error('myfuns:eulerForward:TooManyInputs', ...
                    'requires at most 2 optional inputs');
            end
            % set defaults for optional inputs
            optargs = {[], []};
            optargs(1:numvarargs) = varargin;
            [input_weights, input] = optargs{:};

            if isempty(input_weights) && isempty(input)
                new_hidden = hidden + obj.DeltaTime * obj.ODE(bias, hidden_weights, hidden);
            else
                new_hidden = hidden + obj.DeltaTime * obj.ODE(bias, hidden_weights, hidden, input_weights, input);
            end
        end

        function  new_hidden = rungeKutta(obj, bias, hidden_weights, hidden, varargin)
            numvarargs = length(varargin);
            if numvarargs > 2
                error('myfuns:rungeKutta:TooManyInputs', ...
                    'requires at most 2 optional inputs');
            end
            % set defaults for optional inputs
            optargs = {[], []};
            optargs(1:numvarargs) = varargin;
            [input_weights, input] = optargs{:};
            % RK4 method
            A = [0 0 0 0; 1/2 0 0 0; 0 1/2 0 0; 0 0 1 0];
            b = [1/6 1/3 1/3 1/6];
            m = length(b);
            k = zeros(length(hidden), m);
            for j = 1 : m
                if isempty(input_weights) && isempty(input)
                    k(:,j) = obj.ODE(bias, hidden_weights, hidden + obj.DeltaTime * sum(k * diag(A(j,:)), 2));
                else
                    k(:,j) = obj.ODE(bias, hidden_weights, hidden + obj.DeltaTime * sum(k * diag(A(j,:)), 2), input_weights, input);
                end
            end
            new_hidden = hidden + obj.DeltaTime * sum(k * diag(b), 2);
        end
    end
end

