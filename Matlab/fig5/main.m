% Main program to obtain GC and coherence results, figure 5
%
% Jorge F. Mejias, 2015
%
%   ----------------


format short;
clear all;
close all;clc;
rng(938197);
Nareas2=2;
%Iext is [V1e2 V4e2; V1i2 V4i2; V1e5 V4e5; V1i5 V4i5]



%---
%time series for s=0.05
s=0.05;
Iext=8*[1 1;0 0;1 1;0 0];
X0=trialstat(s,Iext);
par=parameters(s);


%Granger causality analysis:
fs=round(1/(par.binx*par.dt));
f0=granger(X0,100,fs,30,1e4);


%coherence anaysis:
window=1000;overlap=round(0.5*window);freqdisplay=2:2:100;
v10=squeeze(X0(1,:,:));v40=squeeze(X0(2,:,:));
for i=1:size(v10,2)
    [pcoh0(i,:),fcoh0(i,:)]=mscohere(v10(:,i),v40(:,i),window,overlap,freqdisplay,fs);
end



%plots:
grangerplot;


