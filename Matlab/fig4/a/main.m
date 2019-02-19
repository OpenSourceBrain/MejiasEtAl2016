%microstimulation protocol

format short;
clear all;
close all;
clc;

rng(938193);
Gw=1.;s=0.1;   %feedback selectivity
I0=2.*[1 1;0 0;2 2;0 0]; %background to all E populations
Iext=15.*[1 0;0 0;1 0;0 0]; %inject at V1, 15

%at rest:
[fx20,px20,fx50,px50,powerpeak0,fpeak0,meaninput0]=trialstat(I0,s,Gw);
%microstimulation:
[fx2,px2,fx5,px5,powerpeak1,fpeak1,meaninput1]=trialstat(Iext+I0,s,Gw);

%significance:
z1=3;z2=4; % for FF
gamma0=powerpeak0(z1,:);gamma1=powerpeak1(z1,:);
alpha0=powerpeak0(z2,:);alpha1=powerpeak1(z2,:);
[~,pgamma]=ttest(gamma0,gamma1)
[~,palpha]=ttest(alpha0,alpha1)

%equivalent constant input, for comparison:
constantinput=squeeze(meaninput1);
[fx29,px29,fx59,px59,powerpeak9,fpeak9,meaninput9]=trialstat(Iext+I0...
    +constantinput,s,0);



%plot:
fig4a







