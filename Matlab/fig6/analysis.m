% Analysis of the time series for the bilayer cortical model
%
% Jorge F. Mejias, 2014
%

function [pxx,fx,frequency,amplitudeA,amplitudeB,amplitudeC,mfr]=analysis(par,re,minfreq)
  
  
  
  %power for the selected layer:
  %restate=re(1,round((par.transient)/par.dt):end);
  restate=re;
  [pxxp,fx]=periodogram(restate',[],[],1./(par.dt*par.binx));
  pxx=smooth(pxxp,400);
  %mean(restate)
  
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
  



  