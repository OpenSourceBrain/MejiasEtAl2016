% MVGC for the V1-V4 interaction
% ATTENTION: this requires the use of the MVGC Matlab toolbox by Anith Seth, freely available from here:
% https://users.sussex.ac.uk/~lionelb/MVGC/html/mvgchelp.html
%
% The code in this section is an adaptation of his code.
%



function f=granger(X,fmax,fs,momax,fres)

%ntrials   = 10;     % number of trials
%nobs      = 80000;   % number of observations per trial
%nvars     = 2;      % number of variables

regmode   = 'OLS';  % VAR model estimation regression mode ('OLS', 'LWR' or empty for default)
icregmode = 'LWR';  % information criteria regression mode ('OLS', 'LWR' or empty for default)

morder    = 'AIC';  % model order to use ('actual', 'AIC', 'BIC' or supplied numerical value)
%momax     = 60;     % maximum model order for model order estimation (before, it was 20)

acmaxlags = [];   % maximum autocovariance lags (empty for automatic calculation) (before, it was 1000)

tstat     = '';     % statistical test for MVGC:  'F' for Granger's F-test (default) or 'chi2' for Geweke's chi2 test
alpha     = 0.05;   % significance level for significance test
mhtc      = 'FDR';  % multiple hypothesis test correction (see routine 'significance')

%fs        = 10000;    % sample rate (Hz) (before, it was 200)
%fres      = [];     % frequency resolution (empty for automatic calculation)
%fres=1e5;

%seed      = 0;      % random seed (0 for unseeded) 


%--------------------------------------

%now X would be your variable data, with size X(5,1000,10) --> 5 variables, 1000 observations per trial, and 10 trials in total
%X=zeros(nvars,nobs,ntrials);


%----------------------------------------


% Calculate information criteria up to specified maximum model order.

ptic('\n*** tsdata_to_infocrit\n');
[AIC,BIC,moAIC,moBIC] = tsdata_to_infocrit(X,momax,icregmode);
ptoc('*** tsdata_to_infocrit took ');

% Plot information criteria.

%figure(2); clf;
%plot_tsdata([AIC BIC]',{'AIC','BIC'},1/fs);
%title('Model order estimation');

%amo = 10; % actual model order, we don't have it here

fprintf('\nbest model order (AIC) = %d\n',moAIC);
fprintf('best model order (BIC) = %d\n',moBIC);
%fprintf('actual model order     = %d\n',amo);

% Select model order.

%if     strcmpi(morder,'actual')
%    morder = amo;
%    fprintf('\nusing actual model order = %d\n',morder);
if strcmpi(morder,'AIC')
    morder = moAIC;
    fprintf('\nusing AIC best model order = %d\n',morder);
elseif strcmpi(morder,'BIC')
    morder = moBIC;
    fprintf('\nusing BIC best model order = %d\n',morder);
else
    fprintf('\nusing specified model order = %d\n',morder);
end

%----------------------------------------------


% Estimate VAR model of selected order from data.

ptic('\n*** tsdata_to_var... ');
[A,SIG] = tsdata_to_var(X,morder,regmode);
ptoc;

% Check for failed regression

assert(~isbad(A),'VAR estimation failed');

% NOTE: at this point we have a model and are finished with the data! - all
% subsequent calculations work from the estimated VAR parameters A and SIG.


%----------------------------------------------------


% The autocovariance sequence drives many Granger causality calculations (see
% next section). Now we calculate the autocovariance sequence G according to the
% VAR model, to as many lags as it takes to decay to below the numerical
% tolerance level, or to acmaxlags lags if specified (i.e. non-empty).

ptic('*** var_to_autocov... ');
[G,info] = var_to_autocov(A,SIG,acmaxlags);
ptoc;

% The above routine does a LOT of error checking and issues useful diagnostics.
% If there are problems with your data (e.g. non-stationarity, colinearity,
% etc.) there's a good chance it'll show up at this point - and the diagnostics
% may supply useful information as to what went wrong. It is thus essential to
% report and check for errors here.

var_info(info,true); % report results (and bail out on error)


%-------------------------------------------------------


% Calculate spectral pairwise-conditional causalities at given frequency
% resolution - again, this only requires the autocovariance sequence.

ptic('\n*** autocov_to_spwcgc... ');
f= autocov_to_spwcgc(G,fres);
ptoc;
% Check for failed spectral GC calculation

assert(~isbad(f,false),'spectral GC calculation failed');

% Plot spectral causal graph.
%figure(3); clf;
%plot_spw(f,fs,[0,fmax]);






