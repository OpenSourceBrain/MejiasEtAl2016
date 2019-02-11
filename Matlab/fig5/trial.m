% Single trial for the E-I linear model
%
% Jorge F. Mejias, 2014
%

function rate=trial(par,Iext,Nareas,Gw)
	

% we rewrite the par structure into local parameters for readiness:
dt=par.dt;triallength=par.triallength;
transient=par.transient;tau=par.tau;
tstep=par.tstep;tstep2=par.tstep2;
J=par.J;inputbg=par.inputbg;W=par.W;


%we set up the variables for this trial:
totalinput=zeros(4,round(triallength/dt),Nareas);
rate=zeros(4,round(triallength/dt),Nareas);
transfer=zeros(4,Nareas);
xi=normrnd(0,1,4,round(triallength/dt),Nareas); %noise for re2,ri2,re5,ri5 and both areas
input=zeros(4,Nareas);

%first iteration:
rate(:,1,:)=5*(1+tanh(2.*xi(:,1,:))); %between 0 and 10 spikes/s
%Now we start the real simulation:
i=2;
for time=2*dt:dt:triallength
	
  %we compute the inputs to every area:
  for k=1:Nareas
    %total input to the k-th area:
    input(:,k)=inputbg+Iext(:,k);
    totalinput(:,i-1,k)=input(:,k)+J*rate(:,i-1,k);
  end
      
  %interareal projections:
  %FB:
  totalinput(1,i-1,1)=totalinput(1,i-1,1)+Gw*W(1,2,1,2)*rate(3,i-1,2);
  totalinput(3,i-1,1)=totalinput(3,i-1,1)+Gw*W(1,2,2,2)*rate(3,i-1,2);
  
  totalinput(2,i-1,1)=totalinput(2,i-1,1)+Gw*W(1,2,3,2)*rate(3,i-1,2);
  totalinput(4,i-1,1)=totalinput(4,i-1,1)+Gw*W(1,2,4,2)*rate(3,i-1,2);
  %FF:
  totalinput(1,i-1,2)=totalinput(1,i-1,2)+Gw*W(2,1,1,1)*rate(1,i-1,1);
  totalinput(3,i-1,2)=totalinput(3,i-1,2)+Gw*W(2,1,2,1)*rate(1,i-1,1);

  
  for k=1:Nareas
    %input after transfer functions:
    for j=1:4
      transfer(j,k)=totalinput(j,i-1,k)/(1-exp(-totalinput(j,i-1,k)));
    end
  end

  
  %we update the firing rates of all areas:
  for k=1:Nareas
    rate(:,i,k)=rate(:,i-1,k)+tstep.*...
    (-rate(:,i-1,k)+transfer(:,k))+tstep2.*xi(:,i-1,k);
  end
  
  
  
	%index iteration
	i=i+1;
end






