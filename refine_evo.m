%目的：将当前种群在低维空间中进行进化，然后将结果与当前最优解进行比较，得到最终最优解。
%输出 进化后最优的解off_pop
%输入 当前种群，档案的解，维度
function [best_off] =  refine_evo(Pop,Acr,re_D,parm,Incre_learning,B,R1)
N          = size(Pop.decs,1);
Archive    = Acr;
% MCR = zeros(20*re_D,1) + 0.2;
% MF  = zeros(20*re_D,1) + 0.2;
MCR  = parm.MCR;
MF   = parm.MF;
MOP = ones(1,5)/5;

% Generate Lable
for i=1:N
    for j =1:N-1
        Y_lable((i-1)*(N-1)+j,1)=1;
    end
    if(i>1)
        for k=1:i-1
            Y_lable((i-1)*(N-1)+k,1)=2;
        end
    end
end
% tr data
t=0;
for n=1:N
    for  m=1:N
        if(n==m)
        else
            t=t+1;
            Xtr(t,:) = [Pop(n).dec,Pop(m).dec];
        end
    end
end
% % Eigencoordinate System
% [~,I]  = sort(Pop.objs,'ascend');
% num    = N;
% TopDec = Pop(I(1:num)).decs;
% B      = orth(cov(TopDec));
% R1= deal(zeros(num-1));
% R1(logical(eye(num-1))) = rand(1,num-1);
% increment learning
incre_lable         = ones(size(Incre_learning,1),1);
incre_Xtr           = Incre_learning;

% Generate parents
for i=1:N
        Population(i) = Pop(1);
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
CR  = repmat(max(0,min(1,CR)),1,re_D);
F   = min(1,trnd(1,N,1).*sqrt(0.1) + MF(randi(end,N,1)));
while any(F<=0)
    F(F<=0) = min(1,trnd(1,sum(F<=0),1).*sqrt(0.1) + MF(randi(end,sum(F<=0),1)));
end
F  = repmat(F,1,re_D);
OP = arrayfun(@(S)find(rand<=cumsum(MOP),1),1:N);
OP = arrayfun(@(S)find(OP==S),1:length(MOP),'UniformOutput',false);
% Generate offspring
PopDec = Population.decs;
OffDec = PopDec;
OffDec(OP{1},:) = PopDec(OP{1},:) + F(OP{1},:).*(Xp1(OP{1},:)-PopDec(OP{1},:)+Xr1(OP{1},:)-Xr2(OP{1},:));
OffDec(OP{2},:) = PopDec(OP{2},:) + F(OP{2},:).*(Xp1(OP{2},:)-PopDec(OP{2},:)+Xr1(OP{2},:)-Xr3(OP{2},:));
OffDec(OP{3},:) = F(OP{3},:).*(Xr1(OP{3},:)+Xp2(OP{3},:)-Xr3(OP{3},:));
if(size(B,2)== N-1)
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
    p1 = randi(re_D,N,1);
    p2 = arrayfun(@(S)find([rand(1,re_D),2]>CR(S,1),1),1:N);
    for i = 1 : N
        Site = [1:p1(i)-1,p1(i)+p2(i):re_D];
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
RCES_row                       = find(RCES_replace==1);
if(size(RCES_row,1)>=1)
    if(size(RCES_row,1)==1)
        best_off      = SOLUTION(OffDec(RCES_row(1),:));
    else
        for i = 1 : size(RCES_row,1)
          off_decs1(i,:) = OffDec(RCES_row(i),:);
        end
        best_off    =   select(off_decs1,clf);
    end
else
    best_off              = Pop(N);
end
end
function [best_off_re] = select(Pop,model)
N                   = size(Pop,1);
% tr data
t=0;
for n=1:N
    for  m=1:N
        if(n==m)
        else
            t=t+1;
            Xte(t,:) = [Pop(n,:),Pop(m,:)];
        end
    end
end
RCES_pred                       = model.predict(py.numpy.array(Xte));
pred                            = double(RCES_pred)';
for i=1:N
    iss(i,1)     = sum(pred((N-1)*(i-1)+1:(N-1)*i,1),1);
end
m=min(iss(:,1));
[row y]=find(iss(:,1)==m);
if isempty(row)
    OffDec1(1,:)       = Pop(randi(3),:);   
else
    if (size(row,1)>1)
        OffDec1(1,:)   = Pop(row(randi(size(row,1))),:);
    end
    if (size(row,1)==1)
        OffDec1(1,:)   = Pop(row(1),:);
    end
end
best_off_re         = SOLUTION(OffDec1(1,:));
clearvars row
end
