
%figure for the bandpass filter:

figure(1)
subplot(2,2,1)
re5=X(2,:,1);
time=par.dt+par.transient:par.dt:par.triallength;
plot(time,re5,'r-','LineWidth',3)
xlim([10 11]);

%band-pass filter between 8 and 12 Hz (or fpeak +- 2Hz):
subplot(2,2,2)
fs=1/par.dt;
[bf,af] = butter(3,[8 12]/(fs/2),'bandpass');
re5bp=filter(bf,af,re5);
plot(time,re5bp,'LineWidth',3)
xlim([10 11]);

subplot(2,2,3)
[pxx,fx]=periodogram(re5',[],[],1./(par.dt));
loglog(fx',pxx')
xlim([1e-2 1e3]);
ylim([1e-10 1e4]);

subplot(2,2,4)
[pxx,fx]=periodogram(re5bp',[],[],1./(par.dt));
loglog(fx',pxx')
xlim([1e-2 1e3]);
ylim([1e-10 1e4]);








