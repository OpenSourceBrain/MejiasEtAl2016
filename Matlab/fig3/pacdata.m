%DATA ANALYSIS PART:


%load 'pac.mat';
X(1,:)=rate(1,round((dt+transient)/dt):end);
X(2,:)=rate(3,round((dt+transient)/dt):end);


%we find the exact peak frequency on the alpha range:
[~,~,fpeakalpha]=analysis(par,X(2,:),4);
fpeakalpha

%band-pass filter L5 activity:
%3th-order filter between 8 and 12 Hz (or fpeak +- 3Hz):
%fmin=fpeakalpha-3;fmax=fpeakalpha+3;fs=1/dt;
fmin=7;fmax=12;fs=1/dt;
[bf,af] = butter(3,[fmin fmax]/(fs/2),'bandpass');
re5bp=-filtfilt(bf,af,X(2,:));  %simulated LFP




%now we locate N decent peaks along the trial (and well spaced).
%we first divide the time series in N zones, of size tzone:
tzone=4.; %in seconds, before it was 3
tzoneindex=round(tzone/dt); %length of tzone in indices
rest=mod(length(time),tzoneindex);
time5=time(1:end-rest);re5=re5bp(1:end-rest);re2=X(1,1:end-rest);
numberofzones=round(length(re5)/tzoneindex);
zones=reshape(re5',[tzoneindex,numberofzones]); % for re5
zones2=reshape(re2',[tzoneindex,numberofzones]); % for re2
%now we find a prominent peak around the center
%of each zone:
tzi1=round(tzoneindex/2-tzoneindex/4); %bottom limit
tzi2=round(tzoneindex/2+tzoneindex/4); %top limit
t1bottom=tzi1*dt; %this is to detect problems
for i=1:numberofzones
    [alphapeaks(i),aploc(i)]=max(zones(tzi1:tzi2,i));
    aploc(i)=aploc(i)+tzi1;
end
%now we choose a segment of 7 cycles centered on 
%the prominent peak for each zone:
seglength=7/fpeakalpha; % 7 alpha cycles per segment, in seconds
if (seglength/2 >= t1bottom)
    fprintf('Problems with segment window.\n');
end 
segindex=round(0.5*seglength/dt); %segment semi-length in indices
segment=1;segment2=1;clear segment;clear segment2;
k=1;
for i=1:numberofzones
    i;aploc(i);
    segind1=round(aploc(i)-segindex);
    segind2=round(aploc(i)+segindex);
    if alphapeaks(i)>=0. %>0.99 means only segments with large peaks
        segment(:,k)=zones(segind1:segind2,i);
        segment2(:,k)=zones2(segind1:segind2,i);
        k=k+1;
    end
end
numberofzones2=size(segment,2);
%and finally, we get the peak-centered alpha wave by averaging:
alphawaves=mean(segment,2);


%%-----------------------------------------------------------------

%%for L2/3, we make a spectrogram of each segment, and then average the
%%result over segments:

re2full=X(1,:);
%we find the average peak frequency on the gamma range:
[~,~,fpeakgamma]=analysis(par,re2full,30);
fpeakgamma
timewindow=7/fpeakgamma; %window in seconds (5 gamma cycles per window)
window=round(timewindow/dt); %window in units of dt (sample units)
overlap=round(0.95*window);
fs=1/dt; %sample frequency
%freqdisplayed=30:0.1:40; %frequencies computed
freqdisplayed=25:0.25:45;

%we obtain the spectrograms
tt=1;ff=1;pp=1;clear tt;clear ff;clear pp;
for i=1:numberofzones2
    [~,ff,tt,pp(:,:,i)]=spectrogram(segment2(:,i),window,overlap,freqdisplayed,fs,'yaxis');
end

%and finally we average (and apply log transformation for visualization):
gammawaves=mean(pp,3);%gammawaves=10*log10(abs(gammawaves)); %log transf





%plot the figure:

hfig=figure(1);sizefig=800;
set(hfig,'Position',[400,400,1000,1000]);
subplot(3,2,[5 6])
alphatime=1:length(alphawaves);
alphatime=alphatime.*dt-segindex*dt;
plot(alphatime,segment(:,1:100),'Color',[0.7 0.7 0.7],'LineWidth',1);hold on;
plot(alphatime,alphawaves,'b','LineWidth',3);
set(gca,'FontSize',30,'LineWidth',5,'TickLength',[0.02 0.02]);
set(gca, 'Layer','top');
xwin=0.24;xlim([-xwin xwin]);
ylabel('LFP, L5/6');
xlabel('Time relative to alpha peak (s)');
set(gca,'box','off');

%figure:
gtimemid=mean(tt);
gammatime=tt-gtimemid;
subplot(3,2,[1 2 3 4])
surf(gammatime,ff,gammawaves);
set(gca,'FontSize',30,'LineWidth',5,'TickLength',[0.03 0.03]);
set(gca, 'Layer','bottom');
axis xy;view(0,90);
ylabel('Frequency (Hz)');
xlim([-xwin xwin]);
shading interp;
colormap(jet);









