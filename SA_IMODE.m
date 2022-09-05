classdef SA_IMODE_2NP < ALGORITHM
% <single> <real> <large/none> <constrained/none>
% Improved multi-operator differential evolution
% minN  ---   4 --- Minimum population size
% aRate --- 2.6 --- Ratio of archive size to population size

%------------------------------- Reference --------------------------------
% K. M. Sallam, S. M. Elsayed, R. K. Chakrabortty, and M. J. Ryan, Improved
% multi-operator differential evolution algorithm for solving unconstrained
% problems, Proceedings of the IEEE Congress on Evolutionary Computation,
% 2020.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [minN,aRate] = Algorithm.ParameterSet(4,2.6);
            
            %% Generate random population
            Population = Problem.Initialization();
            Archive    = [];
            MCR = zeros(20*Problem.D,1) + 0.2;
            MF  = zeros(20*Problem.D,1) + 0.2;
            k   = 1;
            MOP = ones(1,5)/5;
            cycle = 0;
            %% init dataset
            train_data             = [];
            Incre_learning         = [];
            incre_lable            = [];
            incre_Xtr              = []; 
             [~,init_rank]         = sort(FitnessSingle(Population));
             init_Population       = Population(init_rank(1:Problem.N));
            for i = 1:Problem.N
                train_data{1,i}    = init_Population(i);
            end
            B_cal                  = 1;
            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Reduce the population size
                N          = ceil((minN-Problem.N)*Problem.FE/Problem.maxFE) + Problem.N;
                [~,rank]   = sort(FitnessSingle(Population));
                Population = Population(rank(1:N));
                Archive    = Archive(randperm(end,min(end,ceil(aRate*N))));
               if any(B_cal)
                 % Eigencoordinate System
                [~,I]  = sort(Population.objs,'ascend');
                num    = N;
                TopDec = Population(I(1:num)).decs;
                B      = orth(cov(TopDec));
                R1= deal(zeros(num-1));
                R1(logical(eye(num-1))) = rand(1,num-1);
               else
               end
                % construct train set
                train_set_uni        = [];
                train_set            = [];
                train_set_tem        = [];
                Y_lable              = [];
                Xtr                  = [];
                ind                  = [];
                train_rank           = [];
                for i = 1:N
                    train_set_tem   = [train_data{1,i},train_set_tem];
                end
                train_set_tem       = [Population,train_set_tem];
                [~,ind]             = unique(train_set_tem.objs);
                trainN              = size(ind,1);
                train_set_uni       = train_set_tem(ind(1:trainN));
                [~,train_rank]      = sort(FitnessSingle(train_set_uni));
                train_set           = train_set_uni(train_rank(1:trainN));
               % increment learning
                incre_lable         = ones(size(Incre_learning,1),1);
                incre_Xtr           = Incre_learning;
                % Generate Lable
                for i=1:trainN
                    for j =1:trainN-1
                        Y_lable((i-1)*(trainN-1)+j,1)=1;
                    end
                    if(i>1)
                        for kk=1:i-1
                            Y_lable((i-1)*(trainN-1)+kk,1)=2;
                        end
                    end
                end
               % tr data
                t=0;
                for n=1:trainN
                    for  m=1:trainN
                        if(n==m)
                        else
                            t=t+1;
                            Xtr(t,:) = [train_set(n).dec,train_set(m).dec];
                        end
                    end
                end  
                % Generate parents, CR, F, and operator for each offspring
                Xp1 = Population(ceil(rand(1,N).*max(1,0.25*N))).decs;
                Xp2 = Population(ceil(rand(1,N).*max(2,0.5*N))).decs;
                Xr1 = Population(randi(end,1,N)).decs;
                Xr3 = Population(randi(end,1,N)).decs;
                P   = [Population,Archive];
                Xr2 = P(randi(end,1,N)).decs;
                CR  = randn(N,1).*sqrt(0.1) + MCR(randi(end,N,1));
                CR  = sort(CR);
                CR  = repmat(max(0,min(1,CR)),1,Problem.D);
                F   = min(1,trnd(1,N,1).*sqrt(0.1) + MF(randi(end,N,1)));
                while any(F<=0)
                    F(F<=0) = min(1,trnd(1,sum(F<=0),1).*sqrt(0.1) + MF(randi(end,sum(F<=0),1)));
                end
                F  = repmat(F,1,Problem.D);
                OP = arrayfun(@(S)find(rand<=cumsum(MOP),1),1:N);
                OP = arrayfun(@(S)find(OP==S),1:length(MOP),'UniformOutput',false);
                % Generate offspring
                PopDec = Population.decs;
                OffDec = PopDec;
                OffDec(OP{1},:) = PopDec(OP{1},:) + F(OP{1},:).*(Xp1(OP{1},:)-PopDec(OP{1},:)+Xr1(OP{1},:)-Xr2(OP{1},:));
                OffDec(OP{2},:) = PopDec(OP{2},:) + F(OP{2},:).*(Xp1(OP{2},:)-PopDec(OP{2},:)+Xr1(OP{2},:)-Xr3(OP{2},:));
                OffDec(OP{3},:) = F(OP{3},:).*(Xr1(OP{3},:)+Xp2(OP{3},:)-Xr3(OP{3},:));
                if(size(B,2)== num-1)
                OffDec(OP{4},:) = PopDec(OP{4},:) + F(OP{4},:)*B*R1*B'.*(Xp1(OP{4},:)-PopDec(OP{4},:)+Xr1(OP{4},:)-Xr3(OP{4},:));
                OffDec(OP{5},:) = F(OP{5},:)*B*R1*B'.*(Xr1(OP{5},:)+Xp2(OP{5},:)-Xr3(OP{5},:));      
                else
                OffDec(OP{4},:) = PopDec(OP{4},:) + F(OP{4},:).*(Xp1(OP{4},:)-PopDec(OP{4},:)+Xr1(OP{4},:)-Xr2(OP{4},:));
                OffDec(OP{5},:) = F(OP{5},:).*(Xr1(OP{5},:)+Xp2(OP{5},:)-Xr3(OP{5},:));    
                end
                if rand < 0.4
                    Site = rand(size(CR)) > CR;
                    OffDec(Site) = PopDec(Site);
                else
                    p1 = randi(Problem.D,N,1);
                    p2 = arrayfun(@(S)find([rand(1,Problem.D),2]>CR(S,1),1),1:N);
                    for i = 1 : N
                        Site = [1:p1(i)-1,p1(i)+p2(i):Problem.D];
                        OffDec(i,Site) = PopDec(i,Site);
                    end
                end
                % Generate predict data
                RCES_Xte                       = [Population.decs,OffDec]; 
                % define model
                clf                            = py.sklearn.neighbors.KNeighborsClassifier(int16(3));
                clf.fit(py.numpy.array([Xtr;incre_Xtr]), py.numpy.array([Y_lable;incre_lable]));
                pre_lable                      = clf.predict(py.numpy.array(RCES_Xte));
                RCES_pred                      = double(pre_lable)';    
                RCES_replace                   =  RCES_pred-1;
                Offspring                      = Population;
                RCES_row                       = find(RCES_replace==1);
                for i = 1 : size(RCES_row,1)
                    Population_CES                = SOLUTION(OffDec(RCES_row(i),:));
                    if(FitnessSingle(Population(RCES_row(i)))>FitnessSingle(Population_CES))
                        Offspring(RCES_row(i))         = Population_CES  ;
                    else
                        Incre                     = [Population(RCES_row(i)).dec,Population_CES.dec];
                        Incre_learning            = [Incre_learning;Incre];
                    end
                end
                if( cycle>=10)
                    Offspring                      = SOLUTION(OffDec);
                end
                % Update the population and archive
                delta   = FitnessSingle(Population) - FitnessSingle(Offspring);
                replace = delta > 0;
                Archive = [Archive,Population(replace)];
                Archive = Archive(randperm(end,min(end,ceil(aRate*N))));
                Population(replace) = Offspring(replace);
              %% build and update train_data
                for i = 1:N
                    if(replace(i))
                        train_data{1,i}       = [Population(i),train_data{1,i}];
                        train_data{1,i}       = train_data{1,i}(1:min(end,2));
                    end
                end
                % Update CR, F, and probabilities of operators
                if any(replace)
                    w      = delta(replace)./sum(delta(replace));
                    MCR(k) = (w'*CR(replace,1).^2)./(w'*CR(replace,1));
                    MF(k)  = (w'*F(replace,1).^2)./(w'*F(replace,1));
                    k      = mod(k,length(MCR)) + 1;
                    cycle  = 0;
                    B_cal  = 1;
                else
                    cycle  = cycle+1;
                    MCR(k) = 0.5;
                    MF(k)  = 0.5;
                    B_cal  = 0;
                end
                if any(cellfun(@isempty,OP))
                	MOP = ones(1,5)/5;
                else
                    delta = max(0,delta./abs(FitnessSingle(Population)));
                    MOP = cellfun(@(S)mean(delta(S)),OP);
                    MOP = max(0.1,min(0.9,MOP./sum(MOP)));
                end
                 clearvars RCES_Xte RCES_row replace 
               % refine the best solution
               parm.MF                             = MF;
               parm.MCR                            = MCR;
               if(Problem.FE<1*Problem.maxFE)
                   [~,rank]                        = sort(FitnessSingle(Population));
                   Population                      = Population(rank(1:N));
                   [best_off]                      =  refine_evo(Population,Archive,Problem.D,parm,Incre_learning,B,R1);
                   if( FitnessSingle(Population(1)) - FitnessSingle(best_off)>0)
                       Population(N)               = best_off;
                       Archive                     = [Archive,best_off];
                       Archive                     = Archive(randperm(end,min(end,ceil(aRate*N))));
                   end
               end
            end
        end
    end
end