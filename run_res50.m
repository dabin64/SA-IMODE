clear all
run_N                = 20;
algorithms_name      = {@SA_IMODE};    
problem_name         = {@F1,@F2,@F3,@F4,@F5,@F6,@F7};                         
D_N                  = {50};                                                  
for i=1:length(D_N)
    for j=1:length(algorithms_name)
        for k=1:length(problem_name)
            avg_res_data                = 0;
            for h= 1:run_N
                [Dec,Obj,Con]           = platemo('algorithm',algorithms_name{j},'problem',problem_name{k},'N',5,'M',1,'D',D_N{i},'maxFE',20*D_N{i});
                 med_res_data(h,k)       = min(Obj); 
                avg_res_data            = avg_res_data + min(Obj);
            end
                 output_res(run_N*(k-1)+1:run_N*k,j)  = med_res_data(:,k); 
               output_med_res(k,2*j-1)  = prctile(med_res_data(1:run_N,k),50); 
               output_med_res(k,2*j)    = prctile(med_res_data(1:run_N,k),75) - prctile(med_res_data(1:run_N,k),25);
                 output_avg_res(k,2*j-1) = avg_res_data / run_N;
                output_avg_res(k,2*j)    = std(med_res_data(1:run_N,k));
        end 
    end
    save([num2str(D_N{i}),'sign_res','.mat'],'output_res');    
    save([num2str(D_N{i}),'med_res','.mat'],'output_med_res');
    save([num2str(D_N{i}),'avg_res','.mat'],'output_avg_res');
end
