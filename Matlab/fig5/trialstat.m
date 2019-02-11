%function to obtain the GC and coherence for one value of s
%averaged over trials

function [X]=trialestad(s,Iext)

%parameters:
estadistica=40;
par=parameters(s);
Nareas=2;

nobs=length(par.dt:par.dt:(par.triallength-par.transient));
binx=par.binx;eta=0.2;Gw=1;
X=zeros(Nareas,round(nobs/binx),estadistica);


for j=1:estadistica
    
    simdatos=cell(Nareas,4);
    %real simulation:
    rate=trial(par,Iext,Nareas,Gw);
    
    for i=1:Nareas
        simdatos{i,1}=rate(1,:,i);simdatos{i,2}=rate(2,:,i);
        simdatos{i,3}=rate(3,:,i);simdatos{i,4}=rate(4,:,i);
        %we save the excit. firing rate re(L2/3)+re(L5) for GC analysis:
        X(i,:,j)=rate(1,round((par.dt+par.transient)/par.dt):binx:end,i).*eta+...
        rate(3,round((par.dt+par.transient)/par.dt):binx:end,i).*(1-eta); 
    end

end







