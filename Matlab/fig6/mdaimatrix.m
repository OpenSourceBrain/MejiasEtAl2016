% we compute the mDAI matrix and its correlation with SLN:

function [SLNchain2,mDAIchain2,rho2,pval2]=mdaimatrix(par,ROI,f,slnMat)


Nareas2=length(ROI); %already specified in hierarchy
slnM=slnMat(ROI,ROI);
f2=f;

mDAIchain=zeros(Nareas2*Nareas2,1);
SLNchain=reshape(slnM,[Nareas2*Nareas2,1]);
SLNxDAI=zeros(1,length(f(1,1,:)));

dt=par.binx*par.dt;
frequ=1:1:length(f(1,1,:));
nyq=2*length(f(1,1,:))*dt;
frequ=frequ./nyq;

perf=permute(f2,[2 1 3]);
realDAI=(f2-perf)./(f2+perf);
for i=1:Nareas2
  realDAI(i,i,:)=0;
end

alpharange=find(frequ>6 & frequ<18);
gammarange=find(frequ>30 & frequ<70);
pmDAI1=realDAI(:,:,alpharange);
pmDAI2=realDAI(:,:,gammarange);
mDAI=(mean(pmDAI2,3)-mean(pmDAI1,3))./2;
mDAI(1:Nareas2+1:Nareas2*Nareas2)=0;
mDAIchain=reshape(mDAI,[Nareas2*Nareas2,1]);
SLNchain2=SLNchain(find(SLNchain>0 & SLNchain<1)); %excluding SLN=1
mDAIchain2=mDAIchain(find(SLNchain>0 & SLNchain<1));
[rho2,pval2]=corr(SLNchain2,mDAIchain2,'Type','Spearman');










