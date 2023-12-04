classdef ODESolver
    %ODESOLVER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        ODE
        DeltaTime
    end
    
    methods
        function obj = ODESolver(eps)
            %ODESOLVER Construct an instance of this class
            %   Detailed explanation goes here
            obj.DeltaTime = eps;
        end
        
        function new_state = eulerForward(obj, ODE, state, varargin)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            numvarargs = length(varargin);
            if numvarargs > 1
                error('myfuns:eulerForward:TooManyInputs', ...
                    'requires at most 1 optional inputs');
            end
            % set defaults for optional inputs
            optargs = {[]};
            optargs(1:numvarargs) = varargin;
            data = optargs{:};

            if isempty(data)
                new_state = state + obj.DeltaTime * ODE(state);
            else
                new_state = state + obj.DeltaTime * ODE(state, input);
            end
        end

        function  new_state = rungeKutta(obj, ODE, state, varargin)
            numvarargs = length(varargin);
            if numvarargs > 1
                error('myfuns:rungeKutta:TooManyInputs', ...
                    'requires at most 1 optional inputs');
            end
            % set defaults for optional inputs
            optargs = {[]};
            optargs(1:numvarargs) = varargin;
            data = optargs{:};
            % RK4 method
            A = [0 0 0 0; 1/2 0 0 0; 0 1/2 0 0; 0 0 1 0];
            b = [1/6 1/3 1/3 1/6];
            m = length(b);
            k = zeros(length(state), m);
            for j = 1 : m
                if isempty(data)
                    k(:,j) = ODE(state + obj.DeltaTime * sum(k * diag(A(j,:)), 2));
                else
                    k(:,j) = ODE(state + obj.DeltaTime * sum(k * diag(A(j,:)), 2), data);
                end
            end
            new_state = state + obj.DeltaTime * sum(k * diag(b), 2);
        end
    end
end

