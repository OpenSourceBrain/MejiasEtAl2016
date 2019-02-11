% Main script for simulating a cortical area (such as V1)
% The area contains two local circuits (one for L2/3 and one for L5)
%
% Jorge F. Mejias, 2014
%
%   ----------------


function [fx2,px2,fx5,px5,powerpeak,freqpeak,meaninput]=trialstat(Iexternal,s,Gw)



%parameters:
par=parameters(s);
Nareas=2;

stat=30; %10, or 20
powerpeak=zeros(4,stat);freqpeak=powerpeak;
nobs=length(par.dt:par.dt:(par.triallength-par.transient));
binx=10;eta=0.2; %before, it was eta=0.25
X=zeros(Nareas,round(nobs/binx),stat);


for j=1:stat

%other stuff:
amplitudeA=zeros(2,Nareas);
amplitudeB=zeros(2,Nareas);
amplitudeC=zeros(2,Nareas);
frequency=zeros(2,Nareas);
mfr=zeros(2,Nareas);
simdatos=cell(Nareas,8);


%real simulation:
Iext=Iexternal;
[rate,meaninput]=trial(par,Iext,Nareas,Gw);

k=1;
for i=1:Nareas
  
  simdatos{i,1}=rate(1,:,i);simdatos{i,2}=rate(2,:,i);
  simdatos{i,3}=rate(3,:,i);simdatos{i,4}=rate(4,:,i);
  %we save the excit. firing rate re(L2/3)+re(L5) for GC analysis:
  X(i,:,j)=rate(1,round((par.dt+par.transient)/par.dt):binx:end,i).*eta+...
  rate(3,round((par.dt+par.transient)/par.dt):binx:end,i).*(1-eta);

  %analysis for layer 2/3, first two outputs are pxx and fx:
  [px2(:,i,j),fx2(:,i,j),frequency(1,i),amplitudeA(1,i),...
  amplitudeB(1,i),amplitudeC(1,i),mfr(1,i)]=analysis(par,rate(1,:,i),30.);
  powerpeak(k,j)=amplitudeA(1,i);
  freqpeak(k,j)=frequency(1,i);
  %powerpeak(k,j)=mfr(1,i);
  k=k+1;
  
  %analysis for layer 5:
  [px5(:,i,j),fx5(:,i,j),frequency(2,i),amplitudeA(2,i),...
  amplitudeB(2,i),amplitudeC(2,i),mfr(2,i)]=analysis(par,rate(3,:,i),3.);
  powerpeak(k,j)=amplitudeA(2,i);
  freqpeak(k,j)=frequency(2,i);
  %powerpeak(k,j)=mfr(2,i);
  k=k+1;
  
  %mfr(2,i)
  
end

%plotting:
%fig2

end






