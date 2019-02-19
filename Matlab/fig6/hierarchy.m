%anatomical and functional hierarchy


function [mDAI,X2,X5,realDAI,f,F,fshuf,Fshuf]=hierarchy(indice,Areas,par,ROI,areaList,flnMat,slnMat)


%parameters:
Iexternal=8; %8 is best
Nareas=length(Areas);
Nareas2=length(ROI);

%other initializations: estadistica*2= time (#min)
estadistica=30; %at least, 30
powerpeak=zeros(4,estadistica);
nobs=length(par.dt:par.dt:(par.triallength-par.transient));
binx=par.binx;eta=par.eta;
X=zeros(Nareas,round(nobs/binx),estadistica);

%external input on the full brain network:
Iext=zeros(4,Nareas);
thalamus=6; %6 is best
Iext([1 3],:)=thalamus;
Iext(1,1)=Iext(1,1)+Iexternal;


for j=1:estadistica
    %real simulation:
    rate=trialdelays(j,par,Iext,Nareas);
    for i=1:Nareas
        %we save the excit. firing rate re(L2/3)+re(L5) for GC analysis:
        t0=round((par.dt+par.transient)/par.dt);
        X(i,:,j)=rate(1,t0:binx:end,i).*eta+rate(3,t0:binx:end,i).*(1-eta);
        X2(i,:,j)=rate(1,t0:binx:end,i);
        X5(i,:,j)=rate(3,t0:binx:end,i);
    end
end




%Granger causality analysis of the full network:
fs=round(1/(par.binx*par.dt));momax=30;
[~,c]=intersect(Areas,ROI); %coordinates of ROIs in the 'Areas' subset
Xprov=X(c,:,:);
[f,pval,sig,F,alpha]=granger(Xprov,100,fs,momax,1e4); %before, fres=1e3


%we obtain the shuffled data as a control:
Xs=Xprov;
for i=1:size(Xprov,1)
    Xs(i,:,:)=Xprov(i,:,randperm(size(Xprov,3)));
end
[fshuf,pvals,sigs,Fshuf,alphas]=granger(Xs,100,fs,momax,1e4);

%visualization:
grangerplot8big(indice,par,f,areaList,ROI);


%------------finally, we compute the mDAI values------%

f2=f;
%we compute the DAI:
dt=par.binx*par.dt;
frequ=1:1:length(f(1,1,:));
nyq=2*length(f(1,1,:))*dt;
frequ=frequ./nyq;

perf=permute(f2,[2 1 3]);
realDAI=(f2-perf)./(f2+perf);
for i=1:Nareas2
  realDAI(i,i,:)=0;
end

%we compute the mDAI:
alpharange=find(frequ>6 & frequ<18);
gammarange=find(frequ>30 & frequ<70);
pmDAI1=realDAI(:,:,alpharange);
pmDAI2=realDAI(:,:,gammarange);
mDAI=(mean(pmDAI2,3)-mean(pmDAI1,3))./2;
mDAI(1:Nareas2+1:Nareas2*Nareas2)=0;





