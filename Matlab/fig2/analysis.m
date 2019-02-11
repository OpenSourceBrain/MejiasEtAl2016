

function [pxx,fx,frequency,amplitudeA,amplitudeB,amplitudeC,mfr]=analysis(par,re,minfreq)
  
  
  
  %power for the selected layer:
  restate=re(1,round((par.transient+par.dt)/par.dt):end);
  [pxx1,fx1]=periodogram(restate',[],[],1./(par.dt));
  %mean(restate)
  
  bin=5;%compress the data (pick up one in every 'bin' points):
  remaining=mod(length(pxx1),bin);
  pxx0=pxx1(1:end-remaining,:);fx0=fx1(1:end-remaining,:);
  pxx=binTraces(pxx0,bin)';fx=binTraces(fx0,bin)';
  
  %now the properties we want to track:
  %first, the frequency of the oscillations:
  z=find(fx>minfreq);   %we find peaks above minfreq (3 Hz in L5, 30 Hz in L2/3)
  newpxx=pxx(z);
  newfx=fx(z);
  
  
  %we locate the peaks in the spectrum:
  [pks,loc]=findpeaks(newpxx);
  z=length(loc); %if there's at least one peak:
  if z>=1
    [~,z3]=max(pks);
    frequency=newfx(loc(z3,1));  % location, in Hz, of the highest peak
    amplitudeA=newpxx(loc(z3,1)); % power of the peak in the spectrum
  else
    frequency=0;  %no oscillations
    amplitudeA=0;
  end
  %now the excitatory mean firing rate and the amplitude of oscillations:
  mfr=mean(restate);
  amplitudeB=2*std(restate);
  amplitudeC=max(restate)-min(restate);
  



  