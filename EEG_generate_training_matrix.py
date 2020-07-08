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
	Updated by Solange Cerny [scerny], Aston University
		Adding SEED .MAT data extraction functionality
"""

import os, sys
import time
import numpy as np
from EEG_feature_extraction import generate_feature_vectors_from_samples
from EEG_feature_extraction import generate_feature_vectors_from_samples_v2


def gen_training_matrix(directory_path, output_file, cols_to_ignore):
	"""
	Reads the csv files in directory_path and assembles the training matrix with 
	the features extracted using the functions from EEG_feature_extraction.
	
	Parameters:
		directory_path (str): directory containing the CSV files to process.
		output_file (str): filename for the output file.
		cols_to_ignore (list): list of columns to ignore from the CSV

    Returns:
		numpy.ndarray: 2D matrix containing the data read from the CSV
	
	Author: 
		Original: [lmanso] 
		Updates and documentation: [fcampelo]
	"""
	
	# Initialise return matrix
	FINAL_MATRIX = None
	
	for x in os.listdir(directory_path):

		# Ignore non-CSV files
		if not x.lower().endswith('.csv'):
			continue
		
		# For safety we'll ignore files containing the substring "test". 
		# [Test files should not be in the dataset directory in the first place]
		if 'test' in x.lower():
			continue
		try:
			name, state, _ = x[:-4].split('-')
		except:
			print ('Wrong file name', x)
			sys.exit(-1)
		if state.lower() == 'concentrating':
			state = 2.
		elif state.lower() == 'neutral':
			state = 1.
		elif state.lower() == 'relaxed':
			state = 0.
		else:
			print ('Wrong file name', x)
			sys.exit(-1)
			
		print ('Using file', x)
		full_file_path = directory_path  +   '/'   + x
		generate_feature_vectors_from_samples(file_path = full_file_path, 
												nsamples = 150, 
												period = 1.,
												state = state,
												remove_redundant = True,
												cols_to_ignore = cols_to_ignore,
												output_file = output_file)
		
		# lines are saved in file inside generate_feature_vectors_from_samples
		# as that way it performs faster than vstack (tested on SSD)
		'''
		print ('resulting vector shape for the file', vectors.shape)
		
		
		if FINAL_MATRIX is None:
			FINAL_MATRIX = vectors
		else:
			FINAL_MATRIX = np.vstack( [ FINAL_MATRIX, vectors ] )

	print ('FINAL_MATRIX', FINAL_MATRIX.shape)
	
	# Shuffle rows
	np.random.shuffle(FINAL_MATRIX)
	
	# convert dtype to float (SEED ends up with <U32)
	FINAL_MATRIX = FINAL_MATRIX.astype(float)


	# Save to file
	np.savetxt(output_file, FINAL_MATRIX, delimiter = ',',
			header = ','.join(header), 
			comments = '')
	'''



	# remove_redundancies occurs within external module where file is loaded afterwards
	'''
	if remove_redundant:
		# Remove redundant lag window features
		to_rm = ["lag1_mean_q3_", "lag1_mean_q4_", "lag1_mean_d_q3q4_",
		         "lag1_max_q3_", "lag1_max_q4_", "lag1_max_d_q3q4_",
				 "lag1_min_q3_", "lag1_min_q4_", "lag1_min_d_q3q4_"]
		
		start = time.time() # Performance
		# Remove redundancies
		for i in range(len(to_rm)):
			for j in range(ry.shape[1]):
				rm_str = to_rm[i] + str(j)
				idx = feat_names.index(rm_str)
				feat_names.pop(idx)
				ret = np.delete(ret, idx, axis = 1)
		end = time.time() # Performance
		performance['remove_redundancies'] = end - start # Performance
	'''

	print (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),'Done.')

	return None


def gen_training_matrix_from_seed_prepro(directory_path, output_file, cols_to_ignore):
	"""
	Reads the mat files in directory_path and assembles the training matrix with 
	the features extracted using the functions from EEG_feature_extraction.
	
	Parameters:
		directory_path (str): directory containing the MAT files to process.
		output_file (str): filename for the output file.
		cols_to_ignore (list): list of columns to ignore from the MAT

    Returns:
		numpy.ndarray: 2D matrix containing the data read from the MAT
	
	Author: 
		Original: [scerny] 
	"""
	
	# Initialise return matrix
	FINAL_MATRIX = None
	
	for x in os.listdir(directory_path):

		# Ignore non-MAT files
		if not x.lower().endswith('.mat'):
			continue
		
		if x.lower() == 'label.mat':
			continue
		
		# For safety we'll ignore files containing the substring "test". 
		# [Test files should not be in the dataset directory in the first place]
		if 'test' in x.lower():
			continue

		#try:
		#	name, state, _ = x[:-4].split('-')
		#except:
		#	print ('Wrong file name', x)
		#	sys.exit(-1)
		#if state.lower() == 'concentrating':
		#	state = 2.
		#elif state.lower() == 'neutral':
		#	state = 1.
		#elif state.lower() == 'relaxed':
		#	state = 0.
		#else:
		#	print ('Wrong file name', x)
		#	sys.exit(-1)
			

		# state is handeled inside generate_feature_vectors_from_samples() when dealing with SEED dataset
		state = 0


		print (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),'Using file', x)
		full_file_path = directory_path  +   '/'   + x
		generate_feature_vectors_from_samples_v2(file_path = full_file_path, 
															nsamples = 128, 
															period = 1.,
															state = state,
															remove_redundant = True,
															cols_to_ignore = cols_to_ignore,
															output_file = output_file)

		# lines are saved in file inside generate_feature_vectors_from_samples
		# as that way it performs faster than vstack (tested on SSD)
		'''
		print ('resulting vector shape for the file', vectors.shape)
		
		
		if FINAL_MATRIX is None:
			FINAL_MATRIX = vectors
		else:
			FINAL_MATRIX = np.vstack( [ FINAL_MATRIX, vectors ] )

	print ('FINAL_MATRIX', FINAL_MATRIX.shape)
	
	# Shuffle rows
	np.random.shuffle(FINAL_MATRIX)
	
	# convert dtype to float (SEED ends up with <U32)
	FINAL_MATRIX = FINAL_MATRIX.astype(float)


	# Save to file
	np.savetxt(output_file, FINAL_MATRIX, delimiter = ',',
			header = ','.join(header), 
			comments = '')
	'''



	# remove_redundancies occurs within external module where file is loaded afterwards
	'''
	if remove_redundant:
		# Remove redundant lag window features
		to_rm = ["lag1_mean_q3_", "lag1_mean_q4_", "lag1_mean_d_q3q4_",
		         "lag1_max_q3_", "lag1_max_q4_", "lag1_max_d_q3q4_",
				 "lag1_min_q3_", "lag1_min_q4_", "lag1_min_d_q3q4_"]
		
		start = time.time() # Performance
		# Remove redundancies
		for i in range(len(to_rm)):
			for j in range(ry.shape[1]):
				rm_str = to_rm[i] + str(j)
				idx = feat_names.index(rm_str)
				feat_names.pop(idx)
				ret = np.delete(ret, idx, axis = 1)
		end = time.time() # Performance
		performance['remove_redundancies'] = end - start # Performance
	'''

	print (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),'Done.')

	return None


if __name__ == '__main__':
	"""
	Main function. The parameters for the script are the following:
		[1] directory_path: The directory where the script will look for the files to process.
		[2] output_file: The filename of the generated output file.
	
	ATTENTION: It will ignore the last column of the CSV file. 
	
	Author:
		Original by [lmanso]
		Documentation: [fcampelo]
"""
	if len(sys.argv) < 3:
		print ('arg1: input dir\narg2: output file')
		sys.exit(-1)
	directory_path = sys.argv[1]
	output_file = sys.argv[2]
	gen_training_matrix(directory_path, output_file, cols_to_ignore = -1)
