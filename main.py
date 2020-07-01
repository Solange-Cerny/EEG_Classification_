"""
A quick tutorial.
To run this file in conda, create env as follows:
   conda create -n dissertation -y python=3.7 dask scipy scikit-learn

## Version history:
May 2020:
   Original script by Jodie Ashford [ashfojsm], Aston University
"""

import EEG_generate_training_matrix
import Build_and_test_classifier
import EEG_classifiers
import EEG_preprocessing_utilities
import time
import numpy as np

timestamp = str(int(time.time())) 


# please run only 1 at the time and pay attention to file names
# e.g. _singleFile points to folder with only1 file in it
to_run = [
         #'jodies_prepro',
         #'jodies_classifier',
         #'seed_prepro_4_chanels_singleFile',
         #'seed_prepro_all_chanels_singleFile',
         #'seed_prepro',
         #'remove_redundancies_and_shuffle',
         #'seed_classifier',
         'run_all_classifiers_once'
         ]


# preprocessing

if 'jodies_prepro' in to_run:
   # To create a training matrix from the files provided
   # (See "A Guide to Machine Learning with EEG with a Muse Headband" for more details)

   # TODO change this to the file path where your training data is
   # File path to raw EEG data from Muse headset
   directory_path = r"training_data"

   # Generate training matrix / calculate all features for data
   # TODO name the output file whatever you like
   EEG_generate_training_matrix.gen_training_matrix(
      directory_path=directory_path, 
      cols_to_ignore=-1, 
      output_file="example_training_matrix_"+timestamp+".csv")

   print('Done jodies_prepro')

if 'seed_prepro_4_chanels_singleFile'in to_run:
   directory_path = r"../SEED/Preprocessed_EEG/singleFile"

   # cols_to_ignore=np.s_[5:] removes all columns from 5th index to the end
   EEG_generate_training_matrix.gen_training_matrix_from_seed_prepro(
      directory_path=directory_path, 
      cols_to_ignore=np.s_[5:], 
      output_file="example_training_matrix_4_chanels_singleFile_"+timestamp+".csv")

   print('Done seed_prepro_4_chanels_singleFile')

if 'seed_prepro_all_chanels_singleFile'in to_run:
   directory_path = r"../SEED/Preprocessed_EEG/singleFile"

   EEG_generate_training_matrix.gen_training_matrix_from_seed_prepro(
      directory_path=directory_path, 
      cols_to_ignore=None, 
      output_file="example_training_matrix_all_chanels_singleFile_"+timestamp+".csv")

   print('Done seed_prepro_all_chanels_singleFile')

if 'seed_prepro'in to_run:
   directory_path = r"../SEED/Preprocessed_EEG"

   EEG_generate_training_matrix.gen_training_matrix_from_seed_prepro(
      directory_path=directory_path, 
      cols_to_ignore=None, 
      output_file="example_training_matrix_all_chanels_all_files_"+timestamp+".csv")

   print('Done seed_prepro')


# redundancies removal

if 'remove_redundancies_and_shuffle'in to_run:
   feature_matrix_file="example_training_matrix_all_chanels_all_files_"+timestamp+".csv"
   #feature_matrix_file="example_training_matrix_4_chanels_singleFile_1592332790.csv"
   
   features_to_remove = [
      "lag1_mean_q3_", 
      "lag1_mean_q4_", 
      "lag1_mean_d_q3q4_",
      "lag1_max_q3_", 
      "lag1_max_q4_", 
      "lag1_max_d_q3q4_",
      "lag1_min_q3_", 
      "lag1_min_q4_", 
      "lag1_min_d_q3q4_"]


   #TODO: add shufling functionality inside remove_redundancies_shuffle

   EEG_preprocessing_utilities.remove_redundancies_shuffle(
      feature_matrix_file=feature_matrix_file,
      to_rm=features_to_remove)

   print('Done remove_redundancies_and_shuffle')


# classifiers

if 'jodies_classifier'in to_run:
   # When this script is running you should see an output like this:
   # "Using file x-concentrating-1.csv - resulting vector shape for the file (116, 989)"
   # Your output training matrix csv should look like the example one provided "example_training_matrix.csv"

   # Building and saving a (Sklearn) random forest classifier trained on the features we just extracted
   # TODO change this to the file path where your training matrix is and name clf_output_file
   training_path = r"example_training_matrix_"+timestamp+".csv"
   Build_and_test_classifier.build_classifier(
      training_path=training_path, 
      test_size=0.2, 
      clf_output_file="Random_Forest_Classifier_"+timestamp)
   # Note accuracy is output as it is calculated in
   # 'Build_and_test_classifier.build_classifier() - # Predict on the testing data'

   print('Done jodies_classifier')

if 'seed_classifier' in to_run:
   training_path = "example_training_matrix_all_chanels_singleFile_"+timestamp+".csv"

   Build_and_test_classifier.build_classifier(
      training_path=training_path, 
      test_size=0.2, 
      clf_output_file="Random_Forest_Classifier_SEED_"+timestamp)

   print('Done seed_classifier')

if 'run_all_classifiers_once' in to_run:
   #training_path = "example_training_matrix_all_chanels_singleFile_"+timestamp+".csv"
   training_path = "example_training_matrix_all_chanels_singleFile_1593514608.csv"

   EEG_classifiers.run_svm(
      training_path=training_path, 
      test_size=0.2, 
      clf_output_file="SVM_Classifier_SEED_"+timestamp)

   EEG_classifiers.run_knn(
      training_path=training_path, 
      test_size=0.2, 
      clf_output_file="KNN_Classifier_SEED_"+timestamp)

   EEG_classifiers.run_random_forest(
      training_path=training_path, 
      test_size=0.2, 
      clf_output_file="Random_Forest_Classifier_SEED_"+timestamp)

   EEG_classifiers.run_ada_boost(
      training_path=training_path, 
      test_size=0.2, 
      clf_output_file="Ada_Boost_Classifier_SEED_"+timestamp)

   EEG_classifiers.run_mlp(
      training_path=training_path, 
      test_size=0.2, 
      clf_output_file="MLP_Classifier_SEED_"+timestamp)

   print('Done run_all_classifiers_once')
