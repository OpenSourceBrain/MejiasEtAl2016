%Fig 4: feedback microstimulation protocol
%
% Jorge Mejias, 2015
%


format short;
clear all;
close all;
clc;

rng(938197); %3 or 7
Gw=1.;s=0.1;   %feedback selectivity, s=0.05 or 0.9
Iext=15.*[0 1;0 0;0 1;0 0]; %inject at V4
I0=1*[1 1;0 0;1 1;0 0]; %some background current

%at rest:
[fx20,px20,fx50,px50,powerpeak0,fpeak0,meaninput0]=trialstat(I0,s,Gw);
%microstimulation:
[fx2,px2,fx5,px5,powerpeak1,fpeak1,meaninput1]=trialstat(Iext+I0,s,Gw);
%significance:
z1=1;z2=2; % for FB
gamma0=powerpeak0(z1,:);gamma1=powerpeak1(z1,:);
alpha0=powerpeak0(z2,:);alpha1=powerpeak1(z2,:);
[~,pgamma]=ttest(gamma0,gamma1)
[~,palpha]=ttest(alpha0,alpha1)


%equivalent constant input, for comparison:
constantinput=squeeze(meaninput1);
[fx29,px29,fx59,px59,powerpeak9,fpeak9,meaninput9]=trialstat(Iext+I0+constantinput,s,0);



%plot
fig4b








