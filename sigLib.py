
#!/usr/bin/python
'''
Description  		:	python library for signal processing
Date created 		:   22/08/17
Date modified 		:  	10/05/19



The script requires the following python libraries:
 
 * numpy
 * scipy
 * pandas
 * random
 * matplotlib
 * statsmodels
 * seaborn

'''


import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pylab as pyl
import seaborn as sns
import scipy

#package statsmodels for ARMA
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm


class sigLib():




	def rectify(self, y, type='full-wave'):
		'''rectify some signal.
		
		Arguments:
		---------
		y 		: 	numpy array with signal
		type	:	type of rectifier (full-wave, half-wave)

		Return:
		---------
		yr 		: 	array with rectified signal
		'''

		if type=='full-wave':
			yr = np.abs(y)
		elif type=='half-wave':
			for i in np.where(y<0)[0]:
				yr[i]=0.0 
		return yr


	def zero_crossing(self, y):
		'''compute zero crossing points in a signal.

		Arguments:
		---------
		y 		: 	numpy array with signal
		
		Return:
		---------
		yc 		:	numpy with ones at zero crossing points
		
		'''
		yc = np.zeros(len(y))
		
		y1=np.diff(y)
		for i in np.where(y1==0)[0]:
			yc[i]=1
		return yc


	def btw_low_pass(self, y, cutf, order, btype, hzf=None, sf=None):
		''' filter a signal with scipy.signal.butter

		Arguments:
		--------- 
		y 		:	1D numpy array with signal to filter
		cutfreq :	high cutoff frequency (pi rad/sample or half-cycles/sample)
		order	:	filter order
		btype	: 	type of filter (lowpass, highpass, bandpass, bandstop)
		hzf     : 	cutoff frequency in Hz	
		sf 		: 	sampling frequency

		Return:
		---------
		yf 		: 	1D numpy array with filtered signal
		'''

		if (hzf!=None) & (sf!=None):
			cutf=hzf*2.0/float(sf)

		if cutf>1:
			raise ValueError, 'cutoff frequency must be in the range [-1, 1]. \
			Use parameters hzf (cutoff frequency in Hz) and sf (sampling frequency) instead.'

		b,a = scipy.signal.butter(order, cutf, btype=btype, output='ba')
		yf = scipy.signal.filtfilt(b,a, y)
		return yf  
	

	def root_mean_square(y):
		'''compute root mean square'''
		y1 = np.power(y,2)
		y1 = np.sqrt(np.sum(y1)/float(len(y1)))
		return y1

	def mean_abs_value(self, y):
		'''compute mean absolute value'''
		y1 = np.abs(y)
		y1 = np.sqrt(np.sum(y1)/float(len(y1)))
		return y1 

	def slope_sign_change(self, y):
		'''compute signal slope change'''
		y1 = np.subtract(y[1:-1],y[:-2])*np.subtract(y[1:-1]-y[2:])
		return y1

	def coeff_of_var(self, y):
		'''compute coefficient of variation'''
		y1 = np.std(y)/np.mean(y)
		return y1

	def sig_noise_ratio(self, y):
		'''compute signal to noise ratio'''
		y1 = np.mean(y)/np.std(y)
		return y1

	def moving_average(self, y, n=3):
		'''compute moving average of a signal'''
		y1 = np.cumsum(y, dtype=float)
		y1[n:] = y1[n:]-y1[:-n]
		return y1[n - 1:]/n


	def find_loc_min(self, y):
		'''find local minima or indexes at which signal is min.
		
		Arguments:
		---------
		y 		:	numpy array with signal
		Return:
		---------
		indx 	:	indexes for local minima'''
		if np.round(np.mean(y),0) != 0.0:
			y = y-np.mean(y)

		indx = (np.diff(np.sign(np.diff(y))) > 0).nonzero()[0]
		return indx

	def find_loc_max(self, y):
		'''find local maxima or indexes at which signal is max.
		
		Arguments:
		---------
		y 		:	numpy array with signal
		Return:
		---------
		indx 	:	indexes for local maxima'''
		if np.round(np.mean(y),0) != 0.0:
			y = y-np.mean(y)

		indx = (np.diff(np.sign(np.diff(y))) < 0).nonzero()[0]
		return indx

	def normalize(self, df, columns):
		'''normalize columns of a data frame. This function can deal with NaN and Inf values.
		
		Arguments:
		---------
		df 		:	pandas dataframe
		columns	:	column to normalize

		Return:
		---------
		df1  	:	pandas dataframe normalised
		'''

		df1 = df.copy()
		for col in columns:
			df[col] = (df[col]-np.nanmean(df.ix[(float('+inf')!=abs(df[col])),col]))/np.nanstd(df.ix[(float('+inf')!=abs(df[col])),col])
		return df1

	def autocov(self, y, h, method=1):
		'''calculate autocovariance.

		Arguments:
		----------
		y 		:	1D numpy array with signal
		h		: 	time lag in samples
		method 	: 	method to be used (1; 2)

		Returns: 	
		----------
		out 	: 	autocovariance'''

		if h>len(y):
			raise ValueError, 'h cannot be > of length of y'
			return None
		if h<0:
			raise ValueError, 'h must be positive'
			return None
		
		out=0;
		for i in range(len(y)-h):
			out += (y[i]-np.mean(y))*(y[i+h]-np.mean(y))

		if method==1:
			out=out/len(y)
		elif method==2:
			out = out/(len(y)-h)
		return out

	def autocor(self, y, h, method=1):
		'''autocorrelation (AC), ratio between autocovariance and variance.

		Arguments:
		----------
		y 			: 	1D numpy with signal
		h 			: 	time lag in samples
		method		: 	method to be used (1; 2) 
						method 1 corresponds to statsmodels.tsa.stattools.acf
		Return:
		----------
		out 		: 	1D numpy array with AC'''

		out = self.autocov(y, h, method)/self.autocov(y, 0, method)
		return out

	def partial_autocor(self, y, hMax, method=1):
		'''calculate partial autocorrelation (PAC) for the signal.

		Arguments:
		----------
		y 		: 	1D numpy array with signal
		hMad 	: 	maximum time lab in samples

		Return:
		----------
		out 	: 	1D numpy array with PAC'''

		ac=[]
		out=[]
		for i in range(hMax+1):
			#compute autocorrelation for the first i h-lag
			ac.append(self.autocor(y, i, method))
			#pdb.set_trace()

			x = range(len(ac))
			if len(x)>1:
				#regression (least squares)
				mdl = sm.OLS(ac,x)
				res = mdl.fit()
				out.append(res.params[0])
			else:
				out.append(ac[-1])
		return out


	def plot_xautocxx(self, y, hMax, hMin=0, method=1, type='acor', save=False, plot=False, path=os.getcwd()):
		'''plot AC, ACV or PAC. fomulas for standard errors (SE) taken from: 
		https://uk.mathworks.com/help/econ/autocorrelation-and-partial-autocorrelation.html?requestedDomain=www.mathworks.com


		Arguments:
		----------
		y 		:	numpy array with signal
		hMax	:	max lag
		hMin 	: 	min lag
		method 	: 	method to be used (1; 2)
		type 	: 	what to compute (acor = autocorrelation, acov = autocovariance, pacor = partial acor)
		save 	: 	boolean for saving plot (default = False)
		plot 	:	boolean for plotting (default = False)
		path 	: 	path to output directory 

		Return:
		----------
		y1 		: 	1D numpy with with AC (acor), ACV (acov) or PAC (pacor)'''

		plt.figure()
		out=[]
		if type=='acov':	
			for i in range(hMin,hMax):
				out.append(self.autocov(y, i, method))
				if i==0:
					plt.plot([i, i],[.0,out[-1]], color='blue', lw=1.5)
				else:
					plt.plot([i, i],[.0,out[-1]], color='black', lw=1.5)
				plt.plot(i, out[-1], 'o', color='blue', ms=5)
			plt.ylabel('autocovariance', fontsize=20)

		elif type=='acor':
			for i in range(hMin,hMax):
				out.append(self.autocor(y, i, method))
				if i==0:
					plt.plot([i, i],[.0,out[-1]], color='blue', lw=1.5)
				else:
					plt.plot([i, i],[.0,out[-1]], color='black', lw=1.5)
				plt.plot(i,out[-1], 'o', color='blue', ms=5)

			plt.ylabel('autocorrelation', fontsize=20)
			#standard error
			se = np.sqrt((1+2*np.sum(np.power(out[1:-1],2)))/len(y)) #formula taken from matlab documetation
			#plt.fill_between(np.arange(hMin, hMax, 1), 1.96*se, -1.96*se, color='lightblue', alpha=0.5)
			plt.axhline(1.96*se, linestyle='--', color='lime', lw=1)
			plt.axhline(-1.96*se, linestyle='--', color='lime', lw=1)

		elif type=='pacor':
			out = self.partial_autocor(y, hMax, method)
			for i in range(0,hMax):	
				if i==0:
					plt.plot([i, i],[.0,out[i]], color='blue', lw=1.5)
				else:
					plt.plot([i, i],[.0,out[i]], color='black', lw=1.5)
				plt.plot(i,out[i], 'o', color='blue', markersize=5)
			plt.ylabel('partial autocorrelation', fontsize=20)
			#standard error
			se = np.sqrt(1/float((len(y)-1))) #formula taken from matlab documentation
			#plt.fill_between(np.arange(hMin, hMax, 1), 1.96*se, -1.96*se, color='lightblue', alpha=0.5)
			plt.axhline(1.96*se, linestyle='--', color='lime', lw=1)
			plt.axhline(-1.96*se, linestyle='--', color='lime', lw=1)

		plt.axhline(0.0, color='black', lw=1)
		plt.tick_params(labelsize=20)
		plt.xlabel('lag', fontsize=20)
		plt.xlim([hMin-1, hMax+1])
		plt.ylim([-1.3, 1.3])
		sns.despine()
		if save:
			plt.savefig(path)
		if plot:
			plt.show()
		return out




	def spectral_density(self, y, hMax=10, method=1, plot=True):
		'''calculate the sample spectral density (S) for a discrete time series.
		spectral density is calculated from the autocovariance.

		Arguments:
		---------
		y 		: 	1D numpy array with the signal
		hMax 	: 	maximum lag
		method	: 	method to be used (1; 2)

		Return:
		---------
		out 	: 	1D numpy with spectral density'''


		freq = np.arange(0,.5,.01) #range of freq

		out=[]
		for f in range(len(freq)):
			for i in range(1,len(y)-1):
				o=0
				o += self.autocov(y, i, method)*np.cos(2*np.pi*freq[f]*i)
			out.append(self.autocov(y, 0, method)+2*o)

		if plot:
			plt.figure()
			plt.title('Spectral density')
			plt.plot(out, 'k-', linewidth=0.8)
			plt.ylabel('Amplitude (dB)')
			plt.xlabel('Normalized frequency')
			plt.tight_layout()
			sns.despine()
			plt.show()

		return out

	def power_spectrum(self, y, hMax=10, method=1, plot=False):
		'''calculate the sample power spectrum (P) for a discrete time series.
		power spectrum is calculated from the autocorrelation.
		
		Arguments:
		---------
		y 		: 	1D numpy array with signal
		hMax 	: 	maximum lag
		method	:	method to be used (1; 2)
		plot 	: 	boolean for plotting (default = False)
		
		Return:
		---------
		y1 		:	1D numpy with power spectrum'''

		
		freq = np.arange(0,.5,.01) #range of freq

		y1=[]
		for f in range(len(freq)): 
			o=0
			for i in range(1,len(y)-1):
				o += self.autocor(y, i, method)*np.cos(2*np.pi*freq[f]*i)
			y1.append(1+2*o)

		if plot:
			plt.figure()
			plt.title('Power spectrum')
			plt.plot(y1, 'k-', linewidth=0.8)
			plt.ylabel('Amplitude (dB)')
			plt.xlabel('Normalized frequency')
			plt.tight_layout()
			sns.despine()
			plt.show()
		return y1

	def gen_white_noise(self, mn, sd, samples=1000, plot=True):
		'''generate white noise samples and plot it.

		Arguments:
		---------
		mn 		: 	mean for signal
		sd 		: 	standard deviation for signal
		samples :	number of samples
		plot 	: 	boolean for plotting (default = True)
		Return:
		---------
		y 		: 	numpy array with white noise'''

		np.random.seed(1)
		y = np.random.normal(mn, sd, size=samples)
		if plot: 
			plt.figure()
			plt.title('White noise')
			plt.plot(y)
			plt.show()
		return y


	def gen_random_walk(self, samples=1000, plot=True):
		'''generate random walk sample without a drift.

		Arguments:
		---------
		mn 		: 	mean for signal
		sd 		: 	standard deviation for signal
		samples :	number of samples
		plot 	: 	boolean for plotting (default = True)
		Return:
		---------
		y 		: 	numpy array with white noise'''

		np.random.seed(1)
		y = w = np.random.normal(size=samples)
		for t in range(samples):
			y[t] = y[t-1] + y[t]
		if plot: 
			plt.figure()
			plt.title('Random walk')
			plt.plot(y)
			plt.show()
		return y


	def fit_ARMAX(self, y, order_ar, order_ma, maxLag=30):
		'''fit autoregression moving average (ARMA) model
		NB: NEEDS FIXING..

		Arguments:
		---------
		order_ar	:	order of autoregression (AR) linear model
		order_ma	: 	order of moving average (MA) linear model
		maxlag 		:	maximim lag

		Return:
		---------
		mdl 		: 	model object '''

		if int(np.mean(y)!=0):
			for t in range(len(y)):
				y[t] = y[t]-np.mean(y)

		u = np.random.randn(len(y), 2)

		mdl = smt.ARMA(y, order=(order_ar, order_ma)).fit(maxlag=maxLag, method='mle', trend='nc', exog=u)
		print(mdl.summary())
		return mdl

	def gen_ARMAsample(self, alphas, betas, samples=1000, burn=4000, plot=False):
		'''generate sample based on ARMA coefficients.

		Arguments:
		---------
		alphas	:	1D numpy array with MA coefficients
		betas	:	1D numpy array with MA coefficients
		samples	:	number of samples
		burn 	:	burnin: no idea..

		Return:
		---------
		y 		: 	ARMA sample'''

		# 1D numpy arrays with coeff ready for filtering
		alphas = np.r_[1, alphas]	
		betas = np.r_[1, betas]

		y = smt.arma_generate_sample(ar=alphas, ma=betas, nsample=samples, burnin=burn)
		
		if plot:
			plt.figure()
			plt.title('ARMA sample: RA(%d) MA(%d)' %(len(alphas), len(betas)))
			plt.plot(y, '-k', linewidth=0.7)
			plt.tight_layout()
			plt.show()
		return y



	def filter_ARMA(self, y, alphas, betas, plotSig=False, plotAutocor=False, iSac=None, iEac=None, hMax=30, hMin=0):
		'''Filter signal based on coefficient found by fitting the an autoregression moving average (ARMA) model

		Arguments
		---------
		alphas	:	1D numpy array with alpha coefficients of AR model
		betas	:	1D numpy array with beta coefficients of MA model
		Return
		---------
		y1 	: 	filtered signal: the output is a white random noise signal'''

		# 1D numpy arrays with coeff ready for filtering
		alphas = np.r_[1, -alphas]
		betas = np.r_[1, betas]

		# the signal should have zero mean
		if int(np.mean(y)!=0):
			for t in range(len(y)):
				y[t] = y[t]-np.mean(y)
		 
		AR=[]
		MA=[]

		for i in range(len(alphas), len(y)):
			ar=0
			for a in range(len(alphas)):
				ar += alphas[a]*y[i-a]
			AR.append(ar)

		for j in range(len(betas), len(AR)):
			ma=0
			for b in range(1,len(betas)):
				ma += betas[b]*AR[j-b]
			MA.append(ma)

		y1 = np.subtract(AR[-len(MA):],MA)

		if plotSig:
			fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
			ax1.set_title('Input signal')
			ax1.plot(y, 'k-', linewidth=0.7)
			ax2.set_title('Output whitened signal')
			ax2.plot(y1, 'k-', linewidth=0.7)

			ax1.set_ylabel('amplitude (V)')
			ax2.set_ylabel('amplitude (V)')
			ax2.set_xlabel('time (samples)')
			plt.tight_layout()
			plt.show()

		if plotAutocor:
			if (iSac==None) | (iEac==None):
				raise ValueError, 'No indexes for autocorrelation'

			yAc=[]
			y1Ac=[]
			for i in range(hMin,hMax):
				yAc.append(self.autocor(y[iSac:iEac], i, method=1))
				y1Ac.append(self.autocor(y1[iSac:iEac], i, method=1))

			fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
			for i in range(len(yAc)):
				if i==0:
					ax1.plot([i, i],[.0,yAc[i]], color='lime', linewidth=1.5)
				else:
					ax1.plot([i, i],[.0,yAc[i]], color='grey', linewidth=1.5)
				ax1.plot(i,yAc[i], 'o', color='blue', markersize=5)

			#standard error
			se = np.sqrt((1+2*np.sum(np.power(yAc[1:-1],2)))/len(y))
			ax1.fill_between(np.arange(hMin, hMax, 1), 2*se, -2*se, color='lightblue', alpha=0.5)

			ax1.axhline(0.0, color='grey', linewidth=1)
			ax1.set_xlim([hMin-1, hMax+1])
			ax1.set_ylim([np.min(yAc)-0.5*np.mean(yAc), np.max(yAc)+0.5*np.mean(yAc)])

			for i in range(len(y1Ac)):
				if i==0:
					ax2.plot([i, i],[.0,y1Ac[i]], color='lime', linewidth=1.5)
				else:
					ax2.plot([i, i],[.0,y1Ac[i]], color='grey', linewidth=1.5)
				ax2.plot(i,y1Ac[i], 'o', color='blue', markersize=5)

			#standard error
			se = np.sqrt((1+2*np.sum(np.power(y1Ac[1:-1],2)))/len(y))
			ax2.fill_between(np.arange(hMin, hMax, 1), 2*se, -2*se, color='lightblue', alpha=0.5)
			
			ax2.set_title('Autocorrelation of output whitened signal')
			ax2.axhline(0.0, color='grey', linewidth=1)
			ax2.set_xlim([hMin-1, hMax+1])
			ax2.set_ylim([np.min(y1Ac)-0.5*np.mean(y1Ac), np.max(y1Ac)+0.5*np.mean(y1Ac)])
			ax2.set_xlabel('lag (samples)')
			plt.tight_layout()
			sns.despine()
			plt.show()

		return y1


	def bestfit_ARMA(self, y):
		'''find order for AR and MA models: < Akaike Information Criterion (AIC)
		the signal must be casual, stationary and invertible'''

		best_aic = np.inf 
		best_order = None
		best_mdl = None
		u = np.random.randn(len(y), 2)

		rng = range(5)
		for i in rng:
		    for j in rng:
		        try:
		            tmp_mdl = smt.ARMA(y, order=(i, j)).fit(method='mle', trend='nc', exog=u);
		            tmp_aic = tmp_mdl.aic
		            if tmp_aic < best_aic:
		                best_aic = tmp_aic
		                best_order = (i, j)
		                best_mdl = tmp_mdl
		        except: continue


		print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
		print best_mdl.summary()
		return best_mdl


	def fit_ARMA(self, y, order_ar, order_ma, maxLag=30):
		'''fit autoregression moving average (ARMA) model. 
		this function does not estimate the best coefficients.

		Arguments:
		----------
		y 			:	numpy array with signal
		order_ar	:	order of autoregression (AR) linear model
		order_ma	:	order of moving average (MA) linear model
		maxlag 		:	max lag
		
		Return:
		----------
		mdl			: 	model object '''

		# if the mean of y is != 0, demean signal 
		if int(np.mean(y)!=0):
			for t in range(len(y)):
				y[t] = y[t]-np.mean(y)

		u = np.random.randn(len(y), 2)

		mdl = smt.ARMA(y, order=(order_ar, order_ma)).fit(maxlag=maxLag, method='mle', trend='nc', exog=u)
		print(mdl.summary())
		return mdl


	def despike(self, sig, SDs=3, interp=3, plot=False):
		'''despike signal, by taking the 2nd differential of the signal
		function based on hl_despike (N. Holmes)

		Arguments:
		----------
		sig 	:	1D numpy array
		SDs 	:	number of standard deviations for outlier detection
		interp 	: 	number of data points used in the interpolation
		plot 	: 	boolean for plotting (default = False)
		
		Return:
		----------
		sig 	: 	despiked signal'''
	
		sig=sig.copy()
		sigin=sig.copy()
		dsig=sig.copy()
		dsig=np.diff(dsig)*np.sign(np.diff(dsig)) #1st diff
		dsig=(dsig-np.mean(dsig))/np.std(dsig)
		spikes=np.where(np.abs(dsig)>SDs)[0]+1 #outliers
		spikes1=spikes.copy()
		
		'''
		# This part needs testing..
		pdb.set_trace()
		# Replace extreme values with NaN to avoid using the value during interpolation
		for i in speikes:
			sig.loc[i]=np.nan
		'''

		if plot:
			fig,(ax1,ax2) = plt.subplots(1,2)
			ax1.set_title('raw signal')
			ax1.plot(sig,'k-',lw=1)
			for i in spikes:
				ax1.plot(i,sig[i],'ro', ms=3)
			
			ax2.set_title('diff signal')
			ax2.plot(dsig,'k-',lw=1)
			for i in spikes:
				ax2.plot(i-1,dsig[i-1],'ro', ms=3)
			plt.tight_layout()
			plt.show()

		if len(spikes)>0:
			# Deal first with spikes that last for more than 1 sample
			ranges = sum((list(s) for s in zip(spikes, spikes[1:]) if s[0]+1 != s[1]), [spikes[0]])
			ranges.append(spikes[-1])
			for r in range(len(ranges)/2):
				#r = index range's spike
				if (ranges[1::2][r]-ranges[::2][r]!=0) & (ranges[1::2][r]+1<len(sig)):
					#if close too close to the end (not sure if this line is necessary)
					for i in range(ranges[1::2][r]-ranges[::2][r]+1):
						#i = each spike in the range
						for p in range(1,interp+1): 
							#p values before each spike that will be replaced, starting from [-interp]
							sig[ranges[::2][r]+i-interp+p]=sig[ranges[::2][r]+i-interp]+(sig[ranges[1::2][r]+1]-sig[ranges[::2][r]+i-interp])*float(p)/((interp+1)+(ranges[1::2][r]-ranges[::2][r]))


				# Remove set of spikes from the list
				for j in range(ranges[::2][r],ranges[1::2][r]+1):
					spikes1 = spikes1[spikes1!=j]

			
			# Then fix what is left
			for i in spikes1:
				if i==0:
					sig[0]=sig[1]						#if 1st sample, replace with 2nd
				elif i==1:
					sig[1]=np.mean([sig[0], sig[2]])	#if 2nd, replace with mean of 1st and 3rd
				elif i==2:
					sig[2]=np.mean([sig[1], sig[3]])	#if 3rd, replace with mean of 2nd and 4rd
				elif i+1==len(sig):
					sig[i]=sig[i-1]						#if last, replace with penultimate
				elif i+1<len(sig):
					for p in range(1,interp+1):
						sig[i-interp+p]=sig[i-interp]+(sig[i+1]-sig[i-interp])*float(p)/interp+1
						#otherwise, interpolate the n==interp points around the spike
			
		if plot:
			fig,(ax1,ax2) = plt.subplots(1,2)
			ax1.set_title('input signal')
			ax1.plot(sigin, 'k-',lw=1)
			ax1.set_ylim([np.min(sigin),np.max(sigin)])
			for i in spikes:
				ax1.plot(i,sigin[i],'ro',ms=3)
			
			ax2.set_title('output signal')
			ax2.plot(sig, 'k-',lw=1)
			for i in spikes:
				ax2.plot(i,sig[i],'ro',ms=3)
			ax2.set_ylim([np.min(sigin),np.max(sigin)])
			plt.show()

		return sig

	def R2Z_trans(r):
		'''transform r correlation coefficients into z-Fisher standardized values'''
		z=np.zeros(len(r))
		z = 0.5*(np.log(1+r) - np.log(1-r))
		return z


	def detect_loc_max(self, df, col, sigfreq, sampfreq, window=0.5, plot=False):
		'''detect local maxima in a periodic signal

		Arguments:
		----------
		df 			:	pandas dataframe
		col 		:	column with signal 
		sigfreq		:	frequency for the periodic signal
		sampfreq	:	signal's sampling frequency
		window		: 	window for detection in seconds (default = 0.5 s)
		plot 		: 	boolean for plotting (default = False)

		Returns:
		----------
		y1			: 	1D numpy array of length = len(df) with ones at where signal is max'''

		# detect
		df = df.copy()
		df['max_%s'%col]=np.zeros(len(df))
		df.ix[self.find_loc_max(df[col]),'max_%s'%col]=1
		df.ix[df[col]<0,'max_%s'%col]=0

		# delete extra maxima (function of P12Lib)
		winL=int(sigfreq*sampfreq*window)
		for i in range(winL,len(df),winL):
			df.loc[df[i-winL:i].ix[df[col]<np.mean(df[col]),:].index,'max_%s'%col]=0
			dat = df[i-winL:i].ix[(df['max_%s'%col]==1),:]
			if len(dat)>1:
				df.ix[dat.ix[dat[col]!=dat[col].max(),:].index,'max_%s'%col]=0

		#clean 2 
		#df.loc[df[col]<np.max(df[col])-2*np.std(df[col]),'max_%s'%col]=0

		if plot:
			plt.figure()
			plt.plot(df['time'], df[col], color='black', lw=1)
			for i in df[df['max_%s'%col]==1].index:
				plt.plot(df.ix[i,'time'], df.ix[i,col], 'o', color='red', ms=3)
			plt.ylabel('max_%s'%col)
			plt.xlabel('time')
			plt.tight_layout()
			sns.despine()
			plt.show()

		y1 = df['max_%s'%col].values
		return y1


	def detect_loc_min(self, df, col, sigfreq, sampfreq, window=0.5, plot=False):
		'''detect local minima in a periodic signal.

		Arguments:
		----------
		df 			:	pandas dataframe
		col 		:	column with signal 
		sigfreq		:	frequency for the periodic signal
		sampfreq	:	signal's sampling frequency
		window		: 	window for detection in seconds (default = 0.5 s)
		plot 		: 	boolean for plotting (default = False)

		Returns:
		----------
		y1			: 	1D numpy array of length = len(df) with ones at where signal is min'''

		# detect
		df = df.copy()
		df['min_%s'%col]=np.zeros(len(df))
		df.ix[self.find_loc_min(df[col]),'min_%s'%col]=1
		df.ix[df[col]>0,'min_%s'%col]=0

		# clean (taken from del_extra_max, P12Lib)
		winL=int(sigfreq*sampfreq*window)
		for i in range(winL,len(df),winL):
			df.loc[df[i-winL:i].ix[df[col]>np.mean(df[col]),:].index,'min_%s'%col]=0
			dat = df[i-winL:i].ix[(df['min_%s'%col]==1),:]
			if len(dat)>1:
				df.ix[dat.ix[dat[col]!=dat[col].min(),:].index,'min_%s'%col]=0

		#clean 2 
		#df.loc[df[col]>np.max(df[col])-2*np.std(df[col]),'max_%s'%col]=0

		if plot:
			plt.figure()
			plt.plot(df['time'], df[col], color='black', lw=1)
			for i in df[df['min_%s'%col]==1].index:
				plt.plot(df.ix[i,'time'], df.ix[i,col], 'o', color='red', ms=3)
			plt.ylabel('min_%s'%col)
			plt.xlabel('time')
			plt.tight_layout()
			sns.despine()
			plt.show()

		y1 = df['min_%s'%col].values
		return y1



	def make_meshgrid(self, x, y):
		"""create a mesh of points to plot in

		Arguments:
		----------
		x: data to base x-axis meshgrid on
		y: data to base y-axis meshgrid on
		#h: stepsize for meshgrid, optional

		Returns:
		--------
		xx, yy : ndarray

		Adapted from web resourse: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
		Example: Plot different SVM classifiers in the iris dataset
	    """
		x_margin = 2.0*np.mean(abs(x)); #print(x_margin)
		y_margin = 2.0*np.mean(abs(y)); #print(y_margin)
		x_min, x_max = x.min() - x_margin, x.max() + x_margin
		y_min, y_max = y.min() - y_margin, y.max() + y_margin
		h = np.mean([(x_max-x_min),(y_max-y_min)])/100.0; #print(h)
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
		return xx, yy



	def norm_pdf(self):
		'''normal probability density sample with mean=0 and var=1.'''
		mn=0;
		var=1;
		y=[];
		x = np.linspace(-5,5,1000);

		for i in range(len(x)):
			y.append(1/(var*np.sqrt(2*np.pi))*np.exp(- np.power(x[i]-mn, 2)/(2*var) ))
		return(y)


	def exp_dist(self, lambd=0.5, plot=True):
		'''esponential probability distribution. Same as stats.expon.pdf.

		Arguments:
		----------
		lambd 		: 	scaling factor
		plot 		:	boolean for plotting (default = True)
		Return:
		----------
		y 			: 	1D numpy vector with probability density
		'''

		x  = np.arange(0, 10, 0.1)
		y = lambd * np.exp(-lambd*x)

		if plot: 	
			plt.figure()
			plt.plot(x,y)
			plt.title('Exponential: $\lambda$ =%.2f' %lambd)
			plt.xlabel('x')
			plt.ylabel('pdf')
			plt.show()
		return y
