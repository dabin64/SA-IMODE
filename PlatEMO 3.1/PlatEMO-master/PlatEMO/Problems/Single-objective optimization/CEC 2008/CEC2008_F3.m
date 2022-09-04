classdef CEC2008_F3 < PROBLEM
% <single> <real> <large/none> <expensive/none>
% Shifted Rosenbrock's function

%------------------------------- Reference --------------------------------
% K. Tang, X. Yao, P. N. Suganthan, C. MacNish, Y.-P. Chen, C.-M. Chen, and
% Z. Yang, Benchmark functions for the CEC'2008 special session and
% competition on large scale global optimization, Nature Inspired
% Computation and Applications Laboratory, USTC, China, 2007.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties
        O;  % Optimal decision vector
    end
    methods
        %% Default settings of the problem
        function Setting(obj)
            CallStack = dbstack('-completenames');
            load(fullfile(fileparts(CallStack(1).file),'CEC2008.mat'),'Data');
            obj.O = Data{3};
            obj.M = 1;
            if isempty(obj.D); obj.D = 100; end
            obj.D = min(obj.D,length(obj.O));
            obj.lower    = zeros(1,obj.D) - 100;
            obj.upper    = zeros(1,obj.D) + 100;
            obj.encoding = 'real';
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            Z = PopDec - repmat(obj.O(1:size(PopDec,2)),size(PopDec,1),1);
            PopObj = sum(100*(Z(:,1:end-1).^2-Z(:,2:end)).^2+(Z(:,1:end-1)-1).^2,2);
        end
    end
end