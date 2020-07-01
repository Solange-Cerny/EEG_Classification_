#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
## Version history:

2018:
	Original script by Dr. Luis Manso [lmanso], Aston University
	
2019, June:
	Revised, commented and updated by Dr. Felipe Campelo [fcampelo], Aston University
	(f.campelo@aston.ac.uk / fcampelo@gmail.com)
	
2020, June:
	Added feature_freq_bands() by Solange Cerny [scerny], Aston University
"""

import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
import time
import csv
import os.path
import scipy.io
import json


def matrix_from_csv_file(file_path):
	"""
	Returns the data matrix given the path of a CSV file.
	
	Parameters:
		file_path (str): path for the CSV file with a time stamp in the first column
			and the signals in the subsequent ones.
			Time stamps are in seconds, with millisecond precision

    Returns:
		numpy.ndarray: 2D matrix containing the data read from the CSV
	
	Author: 
		Original: [lmanso] 
		Revision and documentation: [fcampelo]
	
	"""
	
	csv_data = np.genfromtxt(file_path, delimiter = ',')
	full_matrix = csv_data[1:]
	return full_matrix


def matrix_from_mat_file_seed_prepro(file_path):
	'''
	There are fifiteen trials for each experiment. The labels of all trials are 

	1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1, 
	
	where 1 for positive, 0 for neutral and -1 for negative.
	For more detailed description of this dataset, please see http://bcmi.sjtu.edu.cn/~seed/
	'''
	raw_mat_data = scipy.io.loadmat(file_path)

	positive = np.empty((0, 0))
	neutral = np.empty((0, 0))
	negative = np.empty((0, 0))

	# keeping track of timestamps ts
	last_positive_ts = 0
	last_neutral_ts = 0
	last_negative_ts = 0
	
	# adding 10 secs to new experiment timestamps will differentiate experiments from one another
	ts_increment = 10

	desired_data = '_eeg'
	for key in raw_mat_data:
		if desired_data not in key: 
			continue

		if '13' in key or \
		   '11' in key or \
		   '8' in key or \
		   '5' in key or \
		   '2' in key: 
			data, last_neutral_ts = get_data_with_timestamps(raw_mat_data[key], last_neutral_ts + ts_increment)
			if neutral.size == 0:
				neutral = data
			else:
				neutral = np.concatenate((neutral, data))

		elif '15' in key or \
		   '12' in key or \
		   '7' in key or \
		   '4' in key or \
		   '3' in key: 
			data, last_negative_ts = get_data_with_timestamps(raw_mat_data[key], last_negative_ts + ts_increment)
			if negative.size == 0:
				negative = data
			else:
				negative = np.concatenate((negative, data))

		elif '14' in key or \
		   '10' in key or \
		   '9' in key or \
		   '6' in key or \
		   '1' in key: 
			data, last_positive_ts = get_data_with_timestamps(raw_mat_data[key], last_positive_ts + ts_increment)
			if positive.size == 0:
				positive = data
			else:
				positive = np.concatenate((positive, data))

	# check if all lines up correctly
	#np.savetxt("raw_seed_all_djc_eeg__timestamp_test.csv", positive, delimiter=",")

	mat_data_prepro = {}
	mat_data_prepro['1'] = positive
	mat_data_prepro['0'] = neutral
	mat_data_prepro['-1'] = negative

	return mat_data_prepro


def get_data_with_timestamps(seed_data, initial_timestamp):
	freq = 200

	# this deals with float imprecission
	initial_timestamp = round(initial_timestamp)

	# for comparison that all lines up correctly
	#np.savetxt("raw_seed_djc_eeg1_test.csv", seed_data, delimiter=",")

	# transpose the SEED matrix
	seed_data = seed_data.T

	# how many rows (samples) are in the file
	sample_count = seed_data.shape[0]

	# generate timestamps column
	timestamps = np.arange(initial_timestamp, initial_timestamp + (sample_count/freq), 1/freq)

	# ensure timestamps size isn't bigger or smaller
	# this ocassionally ocurs when generating range using np.arange()
	while True:
		if timestamps.shape[0] > sample_count:
			timestamps = np.delete(arr=timestamps, obj=timestamps.shape[0]-1, axis=0)
		elif timestamps.shape[0] < sample_count:
			print ('Wrong timestamp column length')
			sys.exit(-1)
		else:
			break

	# insert timestamps column at the first position (obj=0)
	# use column insertion (axis=1)
	seed_data = np.insert(arr=seed_data, obj=0, values=timestamps, axis=1)

	# check if all lines up correctly
	#np.savetxt("raw_seed_reshaped_djc_eeg1_test.csv", seed_data, delimiter=",")
	#print(seed_data)
	#print(seed_data.shape)

	return seed_data, timestamps[-1]


def get_time_slice(full_matrix, start = 0., period = 1.):
	"""
	Returns a slice of the given matrix, where start is the offset and period is 
	used to specify the length of the signal.
	
	Parameters:
		full_matrix (numpy.ndarray): matrix returned by matrix_from_csv()
		start (float): start point (in seconds after the beginning of records) 
		period (float): duration of the slice to be extracted (in seconds)

	Returns:
		numpy.ndarray: 2D matrix with the desired slice of the matrix
		float: actual length of the resulting time slice
		
	Author:
		Original: [lmanso]
		Reimplemented: [fcampelo]
	"""
	
	# Changed for greater efficiency [fcampelo]
	rstart  = full_matrix[0, 0] + start
	index_0 = np.max(np.where(full_matrix[:, 0] <= rstart))
	index_1 = np.max(np.where(full_matrix[:, 0] <= rstart + period))
	
	duration = full_matrix[index_1, 0] - full_matrix[index_0, 0]
	return full_matrix[index_0:index_1, :], duration


def feature_mean(matrix):
	"""
	Returns the mean value of each signal for the full time window
	
	Parameters:
		matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
		values of nsignals for a time window of length nsamples
		
	Returns:
		numpy.ndarray: 1D array containing the means of each column from the input matrix
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [lmanso]
		Revision and documentation: [fcampelo]
	"""
	
	ret = np.mean(matrix, axis = 0).flatten()
	names = ['mean_' + str(i) for i in range(matrix.shape[1])]
	return ret, names



def feature_mean_d(h1, h2):
	"""
	Computes the change in the means (backward difference) of all signals 
	between the first and second half-windows, mean(h2) - mean(h1)
	
	Parameters:
		h1 (numpy.ndarray): 2D matrix containing the signals for the first 
		half-window
		h2 (numpy.ndarray): 2D matrix containing the signals for the second 
		half-window
		
	Returns:
		numpy.ndarray: 1D array containing the difference between the mean in h2 
		and the mean in h1 of all signals
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [lmanso]
		Revision and documentation: [fcampelo]
	
	"""
	ret = (feature_mean(h2)[0] - feature_mean(h1)[0]).flatten()
	
	
	# Fixed naming [fcampelo]
	names = ['mean_d_h2h1_' + str(i) for i in range(h1.shape[1])]
	return ret, names



def feature_mean_q(q1, q2, q3, q4):
	"""
	Computes the mean values of each signal for each quarter-window, plus the 
	paired differences of means of each signal for the quarter-windows, i.e.,
	feature_mean(q1), feature_mean(q2), feature_mean(q3), feature_mean(q4),
	(feature_mean(q1) - feature_mean(q2)), (feature_mean(q1) - feature_mean(q3)),
	...
	
	Parameters:
		q1 (numpy.ndarray): 2D matrix containing the signals for the first 
		quarter-window
		q2 (numpy.ndarray): 2D matrix containing the signals for the second 
		quarter-window
		q3 (numpy.ndarray): 2D matrix containing the signals for the third 
		quarter-window
		q4 (numpy.ndarray): 2D matrix containing the signals for the fourth 
		quarter-window
		
	Returns:
		numpy.ndarray: 1D array containing the means of each signal in q1, q2, 
		q3 and q4; plus the paired differences of the means of each signal on 
		each quarter-window.
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [lmanso]
		Revision and documentation: [fcampelo]
	
	"""
	v1 = feature_mean(q1)[0]
	v2 = feature_mean(q2)[0]
	v3 = feature_mean(q3)[0]
	v4 = feature_mean(q4)[0]
	ret = np.hstack([v1, v2, v3, v4, 
				     v1 - v2, v1 - v3, v1 - v4, 
					 v2 - v3, v2 - v4, v3 - v4]).flatten()
	
	
	# Fixed naming [fcampelo]
	names = []
	for i in range(4): # for all quarter-windows
		names.extend(['mean_q' + str(i + 1) + "_" + str(j) for j in range(len(v1))])
	
	for i in range(3): # for quarter-windows 1-3
		for j in range((i + 1), 4): # and quarter-windows (i+1)-4
			names.extend(['mean_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k) for k in range(len(v1))])
			 
	return ret, names




def feature_stddev(matrix):
	"""
	Computes the standard deviation of each signal for the full time window
	
	Parameters:
		matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
		values of nsignals for a time window of length nsamples
		
	Returns:
		numpy.ndarray: 1D array containing the standard deviation of each column 
		from the input matrix
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [lmanso]
		Revision and documentation: [fcampelo]
	"""
	
	# fix ddof for finite sampling correction (N-1 instead of N in denominator)
	ret = np.std(matrix, axis = 0, ddof = 1).flatten()
	names = ['std_' + str(i) for i in range(matrix.shape[1])]
	
	return ret, names



def feature_stddev_d(h1, h2):
	"""
	Computes the change in the standard deviations (backward difference) of all 
	signals between the first and second half-windows, std(h2) - std(h1)
	
	Parameters:
		h1 (numpy.ndarray): 2D matrix containing the signals for the first 
		half-window
		h2 (numpy.ndarray): 2D matrix containing the signals for the second 
		half-window
		
	Returns:
		numpy.ndarray: 1D array containing the difference between the stdev in h2 
		and the stdev in h1 of all signals
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [lmanso]
		Revision and documentation: [fcampelo]
	
	"""
	
	ret = (feature_stddev(h2)[0] - feature_stddev(h1)[0]).flatten()
	
	# Fixed naming [fcampelo]
	names = ['std_d_h2h1_' + str(i) for i in range(h1.shape[1])]
	
	return ret, names




def feature_moments(matrix):
	"""
	Computes the 3rd and 4th standardised moments about the mean (i.e., skewness 
	and kurtosis) of each signal, for the full time window. Notice that 
	scipy.stats.moments() returns the CENTRAL moments, which need to be 
	standardised to compute skewness and kurtosis.
	Notice: Kurtosis is calculated as excess kurtosis, e.g., with the Gaussian 
	kurtosis set as the zero point (Fisher's definition)
	- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html
	- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
	- https://en.wikipedia.org/wiki/Standardized_moment
	- http://www.econ.nyu.edu/user/ramseyj/textbook/pg93.99.pdf
	
	Parameters:
		matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
		values of nsignals for a time window of length nsamples
		
	Returns:
		numpy.ndarray: 1D array containing the skewness and kurtosis of each 
		column from the input matrix
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [fcampelo]
	"""

	skw = scipy.stats.skew(matrix, axis = 0, bias = False)
	krt = scipy.stats.kurtosis(matrix, axis = 0, bias = False)
	ret  = np.append(skw, krt)
		
	names = ['skew_' + str(i) for i in range(matrix.shape[1])]
	names.extend(['kurt_' + str(i) for i in range(matrix.shape[1])])
	return ret, names




def feature_max(matrix):
	"""
	Returns the maximum value of each signal for the full time window
	
	Parameters:
		matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
		values of nsignals for a time window of length nsamples
		
	Returns:
		numpy.ndarray: 1D array containing the max of each column from the input matrix
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [lmanso]
		Revision and documentation: [fcampelo]
	"""
	
	ret = np.max(matrix, axis = 0).flatten()
	names = ['max_' + str(i) for i in range(matrix.shape[1])]
	return ret, names



def feature_max_d(h1, h2):
	"""
	Computes the change in max values (backward difference) of all signals 
	between the first and second half-windows, max(h2) - max(h1)
	
	Parameters:
		h1 (numpy.ndarray): 2D matrix containing the signals for the first 
		half-window
		h2 (numpy.ndarray): 2D matrix containing the signals for the second 
		half-window
		
	Returns:
		numpy.ndarray: 1D array containing the difference between the max in h2 
		and the max in h1 of all signals
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [lmanso]
		Revision and documentation: [fcampelo]
	
	"""
	
	ret = (feature_max(h2)[0] - feature_max(h1)[0]).flatten()
	
	# Fixed naming [fcampelo]
	names = ['max_d_h2h1_' + str(i) for i in range(h1.shape[1])]
	return ret, names


def feature_max_q(q1, q2, q3, q4):
	"""
	Computes the max values of each signal for each quarter-window, plus the 
	paired differences of max values of each signal for the quarter-windows, 
	i.e., feature_max(q1), feature_max(q2), feature_max(q3), feature_max(q4),
	(feature_max(q1) - feature_max(q2)), (feature_max(q1) - feature_max(q3)),
	...
	
	Parameters:
		q1 (numpy.ndarray): 2D matrix containing the signals for the first 
		quarter-window
		q2 (numpy.ndarray): 2D matrix containing the signals for the second 
		quarter-window
		q3 (numpy.ndarray): 2D matrix containing the signals for the third 
		quarter-window
		q4 (numpy.ndarray): 2D matrix containing the signals for the fourth 
		quarter-window
		
	Returns:
		numpy.ndarray: 1D array containing the max of each signal in q1, q2, 
		q3 and q4; plus the paired differences of the max values of each signal 
		on each quarter-window.
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [lmanso]
		Revision and documentation: [fcampelo]
	
	"""
	v1 = feature_max(q1)[0]
	v2 = feature_max(q2)[0]
	v3 = feature_max(q3)[0]
	v4 = feature_max(q4)[0]
	ret = np.hstack([v1, v2, v3, v4, 
				     v1 - v2, v1 - v3, v1 - v4, 
					 v2 - v3, v2 - v4, v3 - v4]).flatten()
	
	
	# Fixed naming [fcampelo]
	names = []
	for i in range(4): # for all quarter-windows
		names.extend(['max_q' + str(i + 1) + "_" + str(j) for j in range(len(v1))])
	
	for i in range(3): # for quarter-windows 1-3
		for j in range((i + 1), 4): # and quarter-windows (i+1)-4
			names.extend(['max_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k) for k in range(len(v1))])
			 
	return ret, names


def feature_min(matrix):
	"""
	Returns the minimum value of each signal for the full time window
	
	Parameters:
		matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
		values of nsignals for a time window of length nsamples
		
	Returns:
		numpy.ndarray: 1D array containing the min of each column from the input matrix
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [lmanso]
		Revision and documentation: [fcampelo]
	"""
	
	ret = np.min(matrix, axis = 0).flatten()
	names = ['min_' + str(i) for i in range(matrix.shape[1])]
	return ret, names



def feature_min_d(h1, h2):
	"""
	Computes the change in min values (backward difference) of all signals 
	between the first and second half-windows, min(h2) - min(h1)
	
	Parameters:
		h1 (numpy.ndarray): 2D matrix containing the signals for the first 
		half-window
		h2 (numpy.ndarray): 2D matrix containing the signals for the second 
		half-window
		
	Returns:
		numpy.ndarray: 1D array containing the difference between the min in h2 
		and the min in h1 of all signals
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [lmanso]
		Revision and documentation: [fcampelo]
	
	"""
	
	ret = (feature_min(h2)[0] - feature_min(h1)[0]).flatten()
	
	# Fixed naming [fcampelo]
	names = ['min_d_h2h1_' + str(i) for i in range(h1.shape[1])]
	return ret, names


def feature_min_q(q1, q2, q3, q4):
	"""
	Computes the min values of each signal for each quarter-window, plus the 
	paired differences of min values of each signal for the quarter-windows, 
	i.e., feature_min(q1), feature_min(q2), feature_min(q3), feature_min(q4),
	(feature_min(q1) - feature_min(q2)), (feature_min(q1) - feature_min(q3)),
	...
	
	Parameters:
		q1 (numpy.ndarray): 2D matrix containing the signals for the first 
		quarter-window
		q2 (numpy.ndarray): 2D matrix containing the signals for the second 
		quarter-window
		q3 (numpy.ndarray): 2D matrix containing the signals for the third 
		quarter-window
		q4 (numpy.ndarray): 2D matrix containing the signals for the fourth 
		quarter-window
		
	Returns:
		numpy.ndarray: 1D array containing the min of each signal in q1, q2, 
		q3 and q4; plus the paired differences of the min values of each signal 
		on each quarter-window.
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [lmanso]
		Revision and documentation: [fcampelo]
	
	"""
	v1 = feature_min(q1)[0]
	v2 = feature_min(q2)[0]
	v3 = feature_min(q3)[0]
	v4 = feature_min(q4)[0]
	ret = np.hstack([v1, v2, v3, v4, 
				     v1 - v2, v1 - v3, v1 - v4, 
					 v2 - v3, v2 - v4, v3 - v4]).flatten()
	
	
	# Fixed naming [fcampelo]
	names = []
	for i in range(4): # for all quarter-windows
		names.extend(['min_q' + str(i + 1) + "_" + str(j) for j in range(len(v1))])
	
	for i in range(3): # for quarter-windows 1-3
		for j in range((i + 1), 4): # and quarter-windows (i+1)-4
			names.extend(['min_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k) for k in range(len(v1))])
			 
	return ret, names


def feature_covariance_matrix(matrix):
	"""
	Computes the elements of the covariance matrix of the signals. Since the 
    covariance matrix is symmetric, only the lower triangular elements 
	(including the main diagonal elements, i.e., the variances of eash signal) 
	are returned. 
	
	Parameters:
		matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
		values of nsignals for a time window of length nsamples
		
	Returns:
		numpy.ndarray: 1D array containing the variances and covariances of the 
        signals
		list: list containing feature names for the quantities calculated.
		numpy.ndarray: 2D array containing the actual covariance matrix

	Author:
		Original: [fcampelo]
	"""
    
	covM = np.cov(matrix.T)
	indx = np.triu_indices(covM.shape[0])
	ret  = covM[indx]
	
	names = []
	for i in np.arange(0, covM.shape[1]):
		for j in np.arange(i, covM.shape[1]):
			names.extend(['covM_' + str(i) + '_' + str(j)])
	
	return ret, names, covM


def feature_eigenvalues(covM):
	"""
	Computes the eigenvalues of the covariance matrix passed as the function 
	argument.
	
	Parameters:
		covM (numpy.ndarray): 2D [nsignals x nsignals] covariance matrix of the 
		signals in a time window
		
	Returns:
		numpy.ndarray: 1D array containing the eigenvalues of the covariance 
		matrix
		list: list containing feature names for the quantities calculated.

	Author:
		Original: [lmanso]
		Revision and documentation: [fcampelo]
	"""
	
	ret   = np.linalg.eigvals(covM).flatten()
	names = ['eigenval_' + str(i) for i in range(covM.shape[0])]
	return ret, names


def feature_logcov(covM):
	"""
	Computes the matrix logarithm of the covariance matrix of the signals. 
	Since the matrix is symmetric, only the lower triangular elements 
	(including the main diagonal) are returned. 
	
	In the unlikely case that the matrix logarithm contains complex values the 
	vector of features returned will contain the magnitude of each component 
	(the covariance matrix returned will be in its original form). Complex 
	values should not happen, as the covariance matrix is always symmetric 
	and positive semi-definite, but the guarantee of real-valued features is in 
	place anyway. 
	
	Details:
		The matrix logarithm is defined as the inverse of the matrix 
		exponential. For a matrix B, the matrix exponential is
		
			$ exp(B) = \sum_{r=0}^{\inf} B^r / r! $,
		
		with 
		
			$ B^r = \prod_{i=1}^{r} B / r $.
			
		If covM = exp(B), then B is a matrix logarithm of covM.
	
	Parameters:
		covM (numpy.ndarray): 2D [nsignals x nsignals] covariance matrix of the 
		signals in a time window
		
	Returns:
		numpy.ndarray: 1D array containing the elements of the upper triangular 
		(incl. main diagonal) of the matrix logarithm of the covariance matrix.
		list: list containing feature names for the quantities calculated.
		numpy.ndarray: 2D array containing the matrix logarithm of covM
		

	Author:
		Original: [fcampelo]
	"""
	log_cov = scipy.linalg.logm(covM)
	indx = np.triu_indices(log_cov.shape[0])
	ret  = np.abs(log_cov[indx])
	
	names = []
	for i in np.arange(0, log_cov.shape[1]):
		for j in np.arange(i, log_cov.shape[1]):
			names.extend(['logcovM_' + str(i) + '_' + str(j)])
	
	return ret, names, log_cov



def feature_fft(matrix, period = 1., mains_f = 50., 
				filter_mains = True, filter_DC = True,
				normalise_signals = True,
				ntop = 10, get_power_spectrum = True):
	"""
	Computes the FFT of each signal. 
	
	Parameters:
		matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
		values of nsignals for a time window of length nsamples
		period (float): width (in seconds) of the time window represented by
		matrix
		mains_f (float): the frequency of mains power supply, in Hz.
		filter_mains (bool): should the mains frequency (plus/minus 1Hz) be 
		filtered out?
		filter_DC (bool): should the DC component be removed?
		normalise_signals (bool): should the signals be normalised to the 
		before interval [-1, 1] before computing the FFT?
		ntop (int): how many of the "top N" most energetic frequencies should 
		also be returned (in terms of the value of the frequency, not the power)
		get_power_spectrum (bool): should the full power spectrum of each 
		signal be returned (in terms of magnitude of each frequency component)
		
	Returns:
		numpy.ndarray: 1D array containing the ntop highest-power frequencies 
		for each signal, plus (if get_power_spectrum is True) the magnitude of 
		each frequency component, for all signals.
		list: list containing feature names for the quantities calculated. The 
		names associated with the power spectrum indicate the frequencies down 
		to 1 decimal place.

	Author:
		Original: [fcampelo]
	"""
	
	# Signal properties
	N   = matrix.shape[0] # number of samples
	T = period / N        # Sampling period
	
	# Scale all signals to interval [-1, 1] (if requested)
	if normalise_signals:
		matrix = -1 + 2 * (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
	
	# Compute the (absolute values of the) FFT
	# Extract only the first half of each FFT vector, since all the information
	# is contained there (by construction the FFT returns a symmetric vector).
	fft_values = np.abs(scipy.fftpack.fft(matrix, axis = 0))[0:N//2] * 2 / N
	
	# Compute the corresponding frequencies of the FFT components
	freqs = np.linspace(0.0, 1.0 / (2.0 * T), N//2)
	
	# Remove DC component (if requested)
	if filter_DC:
		fft_values = fft_values[1:]
		freqs = freqs[1:]
		
	# Remove mains frequency component(s) (if requested)
	if filter_mains:
		indx = np.where(np.abs(freqs - mains_f) <= 1)
		fft_values = np.delete(fft_values, indx, axis = 0)
		freqs = np.delete(freqs, indx)
	
	# Extract top N frequencies for each signal
	indx = np.argsort(fft_values, axis = 0)[::-1]
	indx = indx[:ntop]
	
	ret = freqs[indx].flatten(order = 'F')
	
	# Make feature names
	names = []
	for i in np.arange(fft_values.shape[1]):
		names.extend(['topFreq_' + str(j) + "_" + str(i) for j in np.arange(1,ntop+1)])
	
	if (get_power_spectrum):
		ret = np.hstack([ret, fft_values.flatten(order = 'F')])
		
		for i in np.arange(fft_values.shape[1]):
			names.extend(['freq_' + "{:03d}".format(int(j)) + "_" + str(i) for j in 10 * np.round(freqs, 1)])
	
	return ret, names


def calc_feature_vector(matrix, state):
	"""
	Calculates all previously defined features and concatenates everything into 
	a single feature vector.
	
	Parameters:
		matrix (numpy.ndarray): 2D [nsamples x nsignals] matrix containing the 
		values of nsignals for a time window of length nsamples
		state (str): label associated with the time window represented in the 
		matrix.
		
	Returns:
		numpy.ndarray: 1D array containing all features
		list: list containing feature names for the features

	Author:
		Original: [lmanso]
		Updates and documentation: [fcampelo]
	"""
	
	# Extract the half- and quarter-windows
	h1, h2 = np.split(matrix, [ int(matrix.shape[0] / 2) ])
	q1, q2, q3, q4 = np.split(matrix, 
						      [int(0.25 * matrix.shape[0]), 
							   int(0.50 * matrix.shape[0]), 
							   int(0.75 * matrix.shape[0])])

	performance = {} # Performance

	var_names = []	
	
	start = time.time() # Performance
	x, v = feature_mean(matrix)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_mean'] = duration # Performance
	var_names += v
	var_values = x
	
	start = time.time() # Performance
	x, v = feature_mean_d(h1, h2)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_mean_d'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])

	start = time.time() # Performance
	x, v = feature_mean_q(q1, q2, q3, q4)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_mean_q'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])

	start = time.time() # Performance
	x, v = feature_stddev(matrix)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_stddev'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])

	start = time.time() # Performance
	x, v = feature_stddev_d(h1, h2)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_stddev_d'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])

	start = time.time() # Performance
	x, v = feature_moments(matrix)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_moments'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])

	start = time.time() # Performance
	x, v = feature_max(matrix)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_max'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])
	
	start = time.time() # Performance
	x, v = feature_max_d(h1, h2)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_max_d'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])

	start = time.time() # Performance
	x, v = feature_max_q(q1, q2, q3, q4)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_max_q'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])
	
	start = time.time() # Performance
	x, v = feature_min(matrix)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_min'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])

	start = time.time() # Performance
	x, v = feature_min_d(h1, h2)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_min_d'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])

	start = time.time() # Performance
	x, v = feature_min_q(q1, q2, q3, q4)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_min_q'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])

	start = time.time() # Performance
	x, v, covM = feature_covariance_matrix(matrix)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_covariance_matrix'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])
	
	start = time.time() # Performance
	x, v = feature_eigenvalues(covM)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_eigenvalues'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])
	
	start = time.time() # Performance
	x, v, log_cov = feature_logcov(covM)
	end = time.time() # Performance
	duration = end - start # Performance
	performance['feature_logcov'] = duration # Performance
	var_names += v
	var_values = np.hstack([var_values, x])
	
	# excluded on 21-Jun-2020 as per advise of [fcampelo]:
	#   Features calculated by feature_fft() should be excluded from the pool of attributes. 
	#   This function should instead be used as a basis to calculate the power distribution of the 
	#   five frequency *bands* (alpha, beta, gamma, delta and theta) by binning all frequency 
	#   components into these five bands.
	#  
	#start = time.time() # Performance
	#x, v = feature_fft(matrix)
	#end = time.time() # Performance
	#duration = end - start # Performance
	#performance['feature_fft'] = duration # Performance
	#var_names += v
	#var_values = np.hstack([var_values, x])
	
	if state != None:
		var_values = np.hstack([var_values, np.array([state])])
		var_names += ['Label']


	fname = 'feature_stats.csv'

	if os.path.isfile(fname):
		with open(fname, 'a', newline='') as stats_file:
			writer = csv.DictWriter(stats_file, performance.keys())
			writer.writerow(performance)

	else:
		with open(fname, 'w', newline='') as stats_file:
			writer = csv.DictWriter(stats_file, performance.keys())
			writer.writeheader()
			writer.writerow(performance)

	return var_values, var_names


def feature_freq_bands(ffts, names, bins = {
		'delta_l': 0.5,
		'delta_h': 4.0,
		'theta_l': 4.0,
		'theta_h': 8.0,
		'alpha_l': 8.0,
		'alpha_h': 12.0,
		'beta_l':  12.0,
		'beta_h':  35.0,
		'gamma_l': 35.0
	}):
	"""
	Calculates bucket sums for all frequency *bands*  (delta, theta, alpha, beta and gamma) 
	in that order e.g. lowest frequency first.
	
	Parameters:
		ffts (numpy.ndarray): 1D array containing calculated fft values for all channels
		names (list): 1D array containing headers associated wit ffts values in  
		format freq_<XXx>_<ch> where XX.x is frequency with precision of 1 decimal place
		and ch is a channel index. Keeping correct header format is important as it 
		controls feature extraction logic.
		bins (dictionary): dictionary of open interval values for all frequency *bands*

	Returns:
		numpy.ndarray: 1D array containing bucket sums for all channels
		list: list containing feature names for the features

	Author:
		Original: [scerny]
	"""
	current_ch_idx = '0'
	channels = []

	current_ch_x = []
	current_ch_y = []
	current_ch_bins = {
		'delta': [],
		'theta': [],
		'alpha': [],
		'beta':  [],
		'gamma': []
	}

	for idx, sample in enumerate(names):
		splt_sample = sample.split('_')

		if splt_sample[2] != current_ch_idx:
			channels.append([
				current_ch_x, 
				current_ch_y, 
				current_ch_bins,
				{
					'delta': sum(current_ch_bins['delta']),
					'theta': sum(current_ch_bins['theta']),
					'alpha': sum(current_ch_bins['alpha']),
					'beta':  sum(current_ch_bins['beta']),
					'gamma': sum(current_ch_bins['gamma'])
				}
			])
			current_ch_x = []
			current_ch_y = []
			current_ch_bins = {
				'delta': [],
				'theta': [],
				'alpha': [],
				'beta':  [],
				'gamma': []
			}

			current_ch_idx = splt_sample[2]

		x = float(splt_sample[1])/10
		y = ffts[idx]
		current_ch_x.append(x)
		current_ch_y.append(y)

		# binning
		if x > bins['delta_l'] and x <= bins['delta_h']:
			current_ch_bins['delta'].append(y)
		if x > bins['theta_l'] and x <= bins['theta_h']:
			current_ch_bins['theta'].append(y)
		if x > bins['alpha_l'] and x <= bins['alpha_h']:
			current_ch_bins['alpha'].append(y)
		if x > bins['beta_l'] and x <= bins['beta_h']:
			current_ch_bins['beta'].append(y)
		if x > bins['gamma_l']:
			current_ch_bins['gamma'].append(y)

	channels.append([
		current_ch_x, 
		current_ch_y, 
		current_ch_bins,
		{
			'delta': sum(current_ch_bins['delta']),
			'theta': sum(current_ch_bins['theta']),
			'alpha': sum(current_ch_bins['alpha']),
			'beta':  sum(current_ch_bins['beta']),
			'gamma': sum(current_ch_bins['gamma'])
		}
	])

	bands = [] # values
	names = [] # sum_alpha_0

	for idx, channel in enumerate(channels):
		for key, value in channel[3].items():
			bands.append(value)
			names.append('sum_' + key + '_' + str(idx))

	return np.asarray(bands), names


"""
Returns a number of feature vectors from a labeled CSV file, and a CSV header 
corresponding to the features generated.
full_file_path: The path of the file to be read
samples: size of the resampled vector
period: period of the time used to compute feature vectors
state: label for the feature vector
"""
def generate_feature_vectors_from_samples(file_path, nsamples, period, 
										  state = None, 
										  remove_redundant = True,
										  cols_to_ignore = None,
										  output_file = None):
	"""
	Reads data from CSV file in "file_path" and extracts statistical features 
	for each time window of width "period". 
	
	Details:
	Successive time windows overlap by period / 2. All signals are resampled to 
	"nsample" points to maintain consistency. Notice that the removal of 
	redundant features (regulated by "remove_redundant") is based on the 
	feature names - therefore, if the names output by the other functions in 
	this script are changed this routine needs to be revised. 
	
	Currently the redundant features removed from the lag window are, 
	for i in [0, nsignals-1]:
		- mean_q3_i,
		- mean_q4_i, 
		- mean_d_q3q4_i,
		- max_q3_i,
		- max_q4_i, 
		- max_d_q3q4_i,
		- min_q3_i,
		- min_q4_i, 
		- min_d_q3q4_i.
	
	Parameters:
		file_path (str): file path to the CSV file containing the records
		nsamples (int): number of samples to use for each time window. The 
		signals are down/upsampled to nsamples
		period (float): desired width of the time windows, in seconds
		state(str/int/float): label to attribute to the feature vectors
 		remove_redundant (bool): Should redundant features be removed from the 
	    resulting feature vectors (redundant features are those that are 
	    repeated due to the 1/2 period overlap between consecutive windows).
		cols_to_ignore (array): array of columns to ignore from the input matrix
		 
		
	Returns:
		numpy.ndarray: 2D array containing features as columns and time windows 
		as rows.
		list: list containing the feature names

	Author:
		Original: [lmanso]
		Reimplemented: [fcampelo]
		Added SEED dataset support: [scerny]
	"""	

	performance = {} # Performance

	
	dict_matrix = {}

	start = time.time() # Performance
	# Read the matrix from file
	if file_path.lower().endswith('.csv'):
		matrix = matrix_from_csv_file(file_path)
		dict_matrix['default'] = matrix

	elif file_path.lower().endswith('.mat'):
		# This will return array of 3 matrices, one for each state
		# then below  while True: loop will run 3 times for each state
		dict_matrix = matrix_from_mat_file_seed_prepro(file_path)
	end = time.time() # Performance
	performance['load_file_from_hdd'] = end - start # Performance

	for key, matrix in dict_matrix.items():
		if key != 'default':
			# set data label for SEED dataset
			state = key

		# We will start at the very begining of the file
		t = 0.
		
		# No previous vector is available at the start
		previous_vector = None
		
		# Until an exception is raised or a stop condition is met
		while True:
			# Get the next slice from the file (starting at time 't', with a 
			# duration of 'period'
			# If an exception is raised or the slice is not as long as we expected, 
			# return the current data available
			try:
				start = time.time() # Performance
				s, dur = get_time_slice(matrix, start = t, period = period)
				if cols_to_ignore is not None:
					s = np.delete(s, cols_to_ignore, axis = 1)
				end = time.time() # Performance
				performance['get_time_slice'] = end - start # Performance
			except IndexError:
				break
			if len(s) == 0:
				break
			if dur < 0.9 * period:
				break
			
			start = time.time() # Performance
			# Perform the resampling of the vector
			ry, rx = scipy.signal.resample(s[:, 1:], num = nsamples, 
									t = s[:, 0], axis = 0)
			
			# Slide the slice by 1/2 period
			t += 0.5 * period
			end = time.time() # Performance
			performance['resample'] = end - start # Performance
			
			




			# added on 21-Jun-2020 as per [fcampelo]:
			#   Features calculated by feature_fft() should be excluded from the pool of attributes. 
			#   This function should instead be used as a basis to calculate the power distribution of the 
			#   five frequency *bands* (alpha, beta, gamma, delta and theta) by binning all frequency 
			#   components into these five bands.
			#  
			ret, names = feature_fft(ry, period = 1., mains_f = 50., 
									filter_mains = True, filter_DC = True,
									normalise_signals = True,
									ntop = 0, get_power_spectrum = True)


			#timestamp = str(int(time.time()))
			#np.savetxt("_feature_fft____matrix_"+timestamp+".csv", ry.astype(float), delimiter=",")
			#np.savetxt("_feature_fft____ret_"+timestamp+".csv", ret.astype(float), delimiter=",")
			#with open("_feature_fft____names_"+timestamp+".csv", 'w', newline='') as data_file:
			#	writer = csv.writer(data_file)
			#	writer.writerow(names)



			start = time.time() # Performance
			# Compute the feature vector. We will be appending the features of the 
			# current time slice and those of the previous one.
			# If there was no previous vector we just set it and continue 
			# with the next vector.
			#r, headers = calc_feature_vector(ry, state) #TODO: commented by [scerny]


			r, headers = feature_freq_bands(ret, names)
			if state != None:
				r = np.hstack([r, np.array([state])])
				headers += ['Label']


			end = time.time() # Performance
			performance['calc_feature_vector'] = end - start # Performance
			
			if previous_vector is not None:
				start = time.time() # Performance
				# If there is a previous vector, the script concatenates the two 
				# vectors and adds the result to the output matrix
				feature_vector = np.hstack([previous_vector, r])
				end = time.time() # Performance
				performance['hstack'] = end - start # Performance
				
				feat_names = ["lag1_" + s for s in headers[:-1]] + headers

				if os.path.isfile(output_file):
					start = time.time() # Performance
					with open(output_file, 'a', newline='') as data_file:
						writer = csv.writer(data_file)
						writer.writerow(feature_vector)
					end = time.time() # Performance
					performance['vstack-not_instead_file-write'] = end - start # Performance
				else:
					with open(output_file, 'w', newline='') as data_file:
						writer = csv.writer(data_file)
						writer.writerow(feat_names)
						writer.writerow(feature_vector)
					
			# Store the vector of the previous window
			previous_vector = r
			if state is not None:
				# Remove the label (last column) of previous vector
				previous_vector = previous_vector[:-1] 


			fname = 'main_stats.csv'

			if os.path.isfile(fname):
				with open(fname, 'a', newline='') as stats_file:
					writer = csv.DictWriter(stats_file, performance.keys())
					writer.writerow(performance)

			else:
				with open(fname, 'w', newline='') as stats_file:
					writer = csv.DictWriter(stats_file, performance.keys())
					writer.writeheader()
					writer.writerow(performance)


	fname = 'main_stats.csv'

	if os.path.isfile(fname):
		with open(fname, 'a', newline='') as stats_file:
			writer = csv.DictWriter(stats_file, performance.keys())
			writer.writerow(performance)

	else:
		with open(fname, 'w', newline='') as stats_file:
			writer = csv.DictWriter(stats_file, performance.keys())
			writer.writeheader()
			writer.writerow(performance)

	'''
	# Return
	return ret, feat_names
	'''


# ========================================================================
"""
Other notes by [fcampelo]:
1) ENTROPY
Entropy does not make sense for the "continuous" distribution of 
signal values. The closest analogue, Shannon's differential entropy, 
has been shown to be incorrect from a mathematical perspective
(see, https://www.crmarsh.com/static/pdf/Charles_Marsh_Continuous_Entropy.pdf
and https://en.wikipedia.org/wiki/Limiting_density_of_discrete_points )
I could not find an easy way to implement the LDDP here, nor any ready-to-use 
function, so I'm leaving entropy out of the features for now.
A possible alternative would be to calculate the entropy of a histogram of each 
signal. Also something to discuss.

2) CORRELATION
The way the correlations were calculated in the previous script didn't make 
much sense. What was being done was calculating the correlations of 75 pairs of 
vectors, each composed of a single observation of the 5 signals. I cannot think 
of any reason why this would be interesting, or carry any useful information
(simply because the first sample of h1 should be no more related to the first 
sample of h2 than it would be to the one immediately after - or before - it).
A (possibly) more useful information would be the correlations of each of the 
5 signals against each other (maybe they can be correlated under some mental 
states and decorrelated for others)? This is already done by the covariance 
matrix.

3) AUTOCORRELATION
A possibility would be to use the autocorrelation and cross-correlation of 
the signals. Both can be easily calculated, but would result in a massive 
amount of features (e.g., full autocorrelation would yield 2N-1 features per 
signal.). Not sure if we want that, but it's something to consider.

4) TSFRESH
Package tsfresh seemingly has a load of features implemented for time series,
it may be worth exploring.
"""
#