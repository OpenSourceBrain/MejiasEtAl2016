% Single trial for the E-I linear model
% to use this, call it instead of 'trial' in hierarchy.m
% keep in mind that you'll need larger GC tau if you use this.
%
% Jorge F. Mejias, 2014
%
% here we include the inter-areal delays:

function rate=trialdelays(iteration,par,Iext,Nareas)
	

mydeal(par);
compls=ones(Nareas,Nareas)-s;
auxones=ones(4,Nareas);

Wfbe1=s.*Wfb;
Wfbe2=compls.*Wfb;


%we set up the variables for this trial:
totalinput=zeros(4,Nareas);
irate=zeros(4,Nareas);
iratenew=zeros(4,Nareas);
inoise=zeros(4,Nareas);
rate=zeros(4,round(triallength/dt),Nareas);
transfer=zeros(4,Nareas);
xi=normrnd(0,1,4,round(triallength/dt),Nareas); %noise for re2,ri2,re5,ri5
drate1=zeros(Nareas,Nareas);
drate3=zeros(Nareas,Nareas);



%we build the identity-block matrix:
block=cell(Nareas);
for i=1:Nareas
    block{i}=ones(Nareas,1);
end
blockmatrix=blkdiag(block{:});


%first iteration:
rate(:,1,:)=5*(1+tanh(2.*xi(:,1,:))); %between 0 and 10 spikes/s
%Now we start the real simulation:
i=2;
for time=2*dt:dt:triallength
	
  %we set the instantaneous and delayed rates for computations:
  irate(:,:)=rate(:,i-1,:);
  
  %we set the matrix with the delays for this iteration:
  if time>0.2 %little transient (>0.059s or 59ms)
      delaynow=i-delay;
      %for rates from L2/3e
      dprov=squeeze(rate(1,delaynow,:));
      dprov=dprov.*blockmatrix;
      [~,~,dprov2]=find(dprov);
      drate1=reshape(dprov2,Nareas,Nareas);
      
      
      %and for rates from L5e:
      dprov=squeeze(rate(3,delaynow,:));
      dprov=dprov.*blockmatrix;
      [~,~,dprov2]=find(dprov);
      drate3=reshape(dprov2,Nareas,Nareas);
  end
  

  %total input to each area:
  input=inputbg+Iext;
  totalinput(:,:)=input+J*irate(:,:);
  
  %interareal FF projections:
  totalinput(1,:)=totalinput(1,:)+(sum(drate1.*Wff,2))';
  %interareal FB projections:
  totalinput(1,:)=totalinput(1,:)+1.0.*(sum(drate3.*Wfbe1,2))';
  totalinput(2,:)=totalinput(2,:)+0.5.*(sum(drate3.*Wfb,2))';
  totalinput(3,:)=totalinput(3,:)+1.0.*(sum(drate3.*Wfbe2,2))';
  totalinput(4,:)=totalinput(4,:)+0.5.*(sum(drate3.*Wfb,2))';
  
  
  %input after transfer functions:
  transfer(:,:)=totalinput(:,:)./(auxones(:,:)-exp(-1.*totalinput(:,:)));

  %we update the firing rates of all areas:
  inoise(:,:)=xi(:,i-1,:);
  iratenew(:,:)=irate(:,:)+tstep.*...
  (-irate(:,:)+transfer(:,:))+tstep2.*inoise(:,:);
  rate(:,i,:)=iratenew(:,:);

  %index iteration
  i=i+1;
end




