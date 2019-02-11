% Main script for simulating a cortical area (such as V1)
% The area contains two local circuits (one for L2/3 and one for L5)
% This code is for inter-laminar phase-amplitude coupling.
%
% Jorge F. Mejias, 2015
%
%   ----------------

format short;
clear all;
close all;
clc;rng(538193);


%parameters and initial conditions:
par=parameters();mydeal(par);
time=(dt+transient):dt:triallength;
Nareas=1;Gw=0.;

%simulation (just one trial):
%inject at L2/3e and L5, run the trial and save the traces:
Iext=[6;0;8;0];%Iext=[6;0;8;0]; %4,10
rate=trial(par,Iext,Nareas,Gw);

%save pac2.mat;
pacdata;










