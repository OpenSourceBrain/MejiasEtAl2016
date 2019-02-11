% Single trial for the E-I linear model
%
% Jorge F. Mejias, 2014
%

function rate=trial(par,Iext,Nareas,Gw)
	

% we rewrite the par structure into local parameters for readiness:
dt=par.dt;triallength=par.triallength;
transient=par.transient;tau=par.tau;
tstep=par.tstep;tstep2=par.tstep2;
J=par.J;inputbg=par.inputbg;


%we set up the variables for this trial:
totalinput=zeros(4,round(triallength/dt));
rate=zeros(4,round(triallength/dt));
transfer=zeros(4,1);
xi=normrnd(0,1,4,round(triallength/dt)); %noise for re2,ri2,re5,ri5 and both areas
input=zeros(4,1);

%first iteration:
rate(:,1)=5*(1+tanh(2.*xi(:,1,:))); %between 0 and 10 spikes/s
%Now we start the real simulation:
i=2;
for time=2*dt:dt:triallength
	
  %total input:
  input(:)=inputbg+Iext(:);
  totalinput(:,i-1)=input(:)+J*rate(:,i-1);

  
  %input after transfer functions:
  for j=1:4
    transfer(j)=totalinput(j,i-1)/(1-exp(-totalinput(j,i-1)));
  end
  
  %we update the firing rates:
  rate(:,i)=rate(:,i-1)+tstep.*...
  (-rate(:,i-1)+transfer(:))+tstep2.*xi(:,i-1);  
  
  %index iteration
  i=i+1;
end






