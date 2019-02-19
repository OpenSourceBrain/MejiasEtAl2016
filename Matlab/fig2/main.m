% Main code for reproducing Fig 1
%
% Jorge F. Mejias, 2014
%


clear all;
close all;
clc;

%parameters:
s=0; %feedback selectivity
par=parameters(s);
Nareas=1;
estad=10;Imin=0;Istep=2;Imax=6;


%other stuff:
simdatos=cell(estad,8);
Gw=0; %inter-areal projection strength 


ii=1;%contrast value, or Iexternal
for Iexternal=Imin:Istep:Imax


    k=1;%realizations:
    for i=1:estad
    
    %stimulation:
    excitinput=Iexternal.*[1;0;1;0]; %inject at L2e and L5e of the area
    Iext=excitinput;  
    rate=trial(par,Iext,Nareas,Gw);
    
    %analysis:
    simdatos{1,1}=rate(1,:,1);
    %analysis for layer 2/3, first two outputs are pxx and fx:
    [simdatos{k,1},simdatos{k,2}]=analysis(par,rate(1,:,1),10.);
    
    %pxx(k,:)=simdatos{k,1}';
    pxx(k,:)=smooth(simdatos{k,1}',80); %100
    fxx(k,:)=simdatos{k,2}';
    
    %next value of input
    k=k+1;
    end

px(ii,:)=mean(pxx,1);
px2(ii,:)=std(pxx,1);


ii=ii+1;
end %end of contrasts


%correction of the variance values, just in case:
px2(px2<0)=0;


figure2



