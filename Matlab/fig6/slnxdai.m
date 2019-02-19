% we compute SLNxDAI per frequency:


function [frequ,SLNxDAI,DAIfreq]=slnxdai(par,ROI,f,slnMat)

Nareas2=length(ROI);
slnM=slnMat(ROI,ROI);

DAIchain=zeros(Nareas2*Nareas2,1);
SLNchain=reshape(slnM,[Nareas2*Nareas2,1]);
SLNxDAI=zeros(1,length(f(1,1,:)));



for i=1:length(f(1,1,:)) % for all frequencies
    
  %i=3;
  newf=f(1:Nareas2,1:Nareas2,i);
  DAI=(newf-newf')./(newf+newf');
  DAI(1:Nareas2+1:Nareas2*Nareas2)=0;DAIfreq(:,:,i)=DAI;
  DAIchain=reshape(DAI,[Nareas2*Nareas2,1]);
  %we remove the connections with zero strength from the lists:
  SLNchain2=SLNchain(find(SLNchain>0));
  DAIchain2=DAIchain(find(SLNchain>0));
  %we also remove the connections with small functional interactions:
  SLNchain3=SLNchain2(find(SLNchain2<0.99));
  DAIchain3=DAIchain2(find(SLNchain2<0.99));
  %correlations:
  [rho2,pval2]=corr(SLNchain3,DAIchain3,'Type','Spearman');
  SLNxDAI(1,i)=mean2(rho2);
  
end


dt=par.binx*par.dt;
frequ=1:1:length(f(1,1,:));
nyq=2*length(f(1,1,:))*dt;
frequ=frequ./nyq;




  
  
  
  
  

