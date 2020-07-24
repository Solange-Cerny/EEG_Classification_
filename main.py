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
import csv
import os.path
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
         #'seed_prepro_all_chanels_singlePerson',
         #'run_all_classifiers_once_1_subject',
         #'seed_prepro_all_chanels_allPeople',
         'run_all_classifiers_once_all_subjects'
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

if 'seed_prepro_all_chanels_singlePerson'in to_run:
   dir_path_train = r"../SEED/Preprocessed_EEG/subject_1_training"
   dir_path_test = r"../SEED/Preprocessed_EEG/subject_1_test"
   
   train_fname = "train_matrix.csv"
   test_fname = "test_matrix.csv"

   EEG_generate_training_matrix.gen_training_matrix_from_seed_prepro(
      directory_path=dir_path_train, 
      cols_to_ignore=None, 
      output_file= dir_path_train + "/" + train_fname)

   EEG_generate_training_matrix.gen_training_matrix_from_seed_prepro(
      directory_path=dir_path_test, 
      cols_to_ignore=None, 
      output_file= dir_path_test + "/" + test_fname)

   print('Done seed_prepro_all_chanels_singlePerson')

if 'seed_prepro_all_chanels_allPeople'in to_run:
   subject_files = [['../SEED/Preprocessed_EEG/subject_1_training',
                     '../SEED/Preprocessed_EEG/subject_1_test'],
                    ['../SEED/Preprocessed_EEG/subject_2_training',
                     '../SEED/Preprocessed_EEG/subject_2_test'],
                    ['../SEED/Preprocessed_EEG/subject_3_training',
                     '../SEED/Preprocessed_EEG/subject_3_test'],
                    ['../SEED/Preprocessed_EEG/subject_4_training',
                     '../SEED/Preprocessed_EEG/subject_4_test'],
                    ['../SEED/Preprocessed_EEG/subject_5_training',
                     '../SEED/Preprocessed_EEG/subject_5_test'],
                    ['../SEED/Preprocessed_EEG/subject_6_training',
                     '../SEED/Preprocessed_EEG/subject_6_test'],
                    ['../SEED/Preprocessed_EEG/subject_7_training',
                     '../SEED/Preprocessed_EEG/subject_7_test'],
                    ['../SEED/Preprocessed_EEG/subject_8_training',
                     '../SEED/Preprocessed_EEG/subject_8_test'],
                    ['../SEED/Preprocessed_EEG/subject_9_training',
                     '../SEED/Preprocessed_EEG/subject_9_test'],
                    ['../SEED/Preprocessed_EEG/subject_10_training',
                     '../SEED/Preprocessed_EEG/subject_10_test'],
                    ['../SEED/Preprocessed_EEG/subject_11_training',
                     '../SEED/Preprocessed_EEG/subject_11_test'],
                    ['../SEED/Preprocessed_EEG/subject_12_training',
                     '../SEED/Preprocessed_EEG/subject_12_test'],
                    ['../SEED/Preprocessed_EEG/subject_13_training',
                     '../SEED/Preprocessed_EEG/subject_13_test'],
                    ['../SEED/Preprocessed_EEG/subject_14_training',
                     '../SEED/Preprocessed_EEG/subject_14_test'],
                    ['../SEED/Preprocessed_EEG/subject_15_training',
                     '../SEED/Preprocessed_EEG/subject_15_test']]

   iter = 1
   shuffle = True

   for subject in subject_files:
      print('Loading ' + str(subject))
   
      train_fname = "train_matrix.csv"
      test_fname = "test_matrix.csv"

      EEG_generate_training_matrix.gen_training_matrix_from_seed_prepro(
         directory_path=subject[0], 
         cols_to_ignore=None, 
         output_file= subject[0] + "/" + train_fname)

      EEG_generate_training_matrix.gen_training_matrix_from_seed_prepro(
         directory_path=subject[1], 
         cols_to_ignore=None, 
         output_file= subject[1] + "/" + test_fname)

   print('Done seed_prepro_all_chanels_allPeople')


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

if 'run_all_classifiers_once_1_subject' in to_run:
   #training_path = "example_training_matrix_all_chanels_singleFile_"+timestamp+".csv"
   training_path = "../SEED/Preprocessed_EEG/subject_1_training/train_matrix.csv"
   test_path = "../SEED/Preprocessed_EEG/subject_1_test/test_matrix.csv"
   resuls_file = "../SEED/Preprocessed_EEG/subject_1_training/results_"+timestamp+".csv"

   iter = 1
   shuffle = True

   #x_train, x_test, y_train, y_test = EEG_classifiers.load_data_one_file_split(
   #   training_path=training_path, 
   #   test_size=0.2, 
   #   random_seed=iter, 
   #   shuffle=shuffle)

   x_train, x_test, y_train, y_test = EEG_classifiers.load_data_two_files(
      training_path=training_path, 
      test_path=test_path)

   mlp_acc, mlp_mcc, mlp_auc = EEG_classifiers.run_mlp(
      "MLP_Classifier_SEED_"+timestamp,
      x_train, x_test, y_train, y_test)

   svm_acc, svm_mcc, svm_auc = EEG_classifiers.run_svm(
      "SVM_Classifier_SEED_"+timestamp,
      x_train, x_test, y_train, y_test)

   knn_acc, knn_mcc, knn_auc = EEG_classifiers.run_knn(
      "KNN_Classifier_SEED_"+timestamp,
      x_train, x_test, y_train, y_test)

   randf_acc, randf_mcc, randf_auc = EEG_classifiers.run_random_forest(
      "Random_Forest_Classifier_SEED_"+timestamp,
      x_train, x_test, y_train, y_test)

   adab_acc, adab_mcc, adab_auc = EEG_classifiers.run_ada_boost(
      "Ada_Boost_Classifier_SEED_"+timestamp,
      x_train, x_test, y_train, y_test)

   # CSV table for single run
   ##########################
   with open(resuls_file, 'w') as f:
      f.write('Classifier,acc,mcc,auc\n')
      f.write('SVM,' + str(svm_acc) + ',' + str(svm_mcc) + ',' + str(svm_auc) + '\n')
      f.write('KNN,' + str(knn_acc) + ',' + str(knn_mcc) + ',' + str(knn_auc) + '\n')
      f.write('Random Forest,' + str(randf_acc) + ',' + str(randf_mcc) + ',' + str(randf_auc) + '\n')
      f.write('ADA B,' + str(adab_acc) + ',' + str(adab_mcc) + ',' + str(adab_auc) + '\n')
      f.write('MLP,' + str(mlp_acc) + ',' + str(mlp_mcc) + ',' + str(mlp_auc) + '\n')

   # CSV table for multi run
   #########################
   if os.path.isfile(resuls_file + '_multi.csv'):
      with open(resuls_file + '_multi.csv', 'a') as f:
         f.write(str(svm_acc) + ',' + str(svm_mcc) + ',' + str(svm_auc) + ','
            + str(knn_acc) + ',' + str(knn_mcc) + ',' + str(knn_auc) + ','
            + str(randf_acc) + ',' + str(randf_mcc) + ',' + str(randf_auc) + ','
            + str(adab_acc) + ',' + str(adab_mcc) + ',' + str(adab_auc) + ','
            + str(mlp_acc) + ',' + str(mlp_mcc) + ',' + str(mlp_auc) + '\n')
   else:
      with open(resuls_file + '_multi.csv', 'w') as f:
         f.write('svm_acc, svm_mcc, svm_auc, '
            + 'knn_acc, knn_mcc, knn_auc, '
            + 'randf_acc, randf_mcc, '
            + 'randf_auc, adab_acc, adab_mcc, '
            + 'adab_auc, mlp_acc, mlp_mcc, mlp_auc\n')
         f.write(str(svm_acc) + ',' + str(svm_mcc) + ',' + str(svm_auc) + ','
            + str(knn_acc) + ',' + str(knn_mcc) + ',' + str(knn_auc) + ','
            + str(randf_acc) + ',' + str(randf_mcc) + ',' + str(randf_auc) + ','
            + str(adab_acc) + ',' + str(adab_mcc) + ',' + str(adab_auc) + ','
            + str(mlp_acc) + ',' + str(mlp_mcc) + ',' + str(mlp_auc) + '\n')

   print('Done run_all_classifiers_once_1_subject')

if 'run_all_classifiers_once_all_subjects' in to_run:
   subject_files = [['../SEED/Preprocessed_EEG/subject_1_training',
                     '../SEED/Preprocessed_EEG/subject_1_test'],
                    ['../SEED/Preprocessed_EEG/subject_2_training',
                     '../SEED/Preprocessed_EEG/subject_2_test'],
                    ['../SEED/Preprocessed_EEG/subject_3_training',
                     '../SEED/Preprocessed_EEG/subject_3_test'],
                    ['../SEED/Preprocessed_EEG/subject_4_training',
                     '../SEED/Preprocessed_EEG/subject_4_test'],
                    ['../SEED/Preprocessed_EEG/subject_5_training',
                     '../SEED/Preprocessed_EEG/subject_5_test'],
                    ['../SEED/Preprocessed_EEG/subject_6_training',
                     '../SEED/Preprocessed_EEG/subject_6_test'],
                    ['../SEED/Preprocessed_EEG/subject_7_training',
                     '../SEED/Preprocessed_EEG/subject_7_test'],
                    ['../SEED/Preprocessed_EEG/subject_8_training',
                     '../SEED/Preprocessed_EEG/subject_8_test'],
                    ['../SEED/Preprocessed_EEG/subject_9_training',
                     '../SEED/Preprocessed_EEG/subject_9_test'],
                    ['../SEED/Preprocessed_EEG/subject_10_training',
                     '../SEED/Preprocessed_EEG/subject_10_test'],
                    ['../SEED/Preprocessed_EEG/subject_11_training',
                     '../SEED/Preprocessed_EEG/subject_11_test'],
                    ['../SEED/Preprocessed_EEG/subject_12_training',
                     '../SEED/Preprocessed_EEG/subject_12_test'],
                    ['../SEED/Preprocessed_EEG/subject_13_training',
                     '../SEED/Preprocessed_EEG/subject_13_test'],
                    ['../SEED/Preprocessed_EEG/subject_14_training',
                     '../SEED/Preprocessed_EEG/subject_14_test'],
                    ['../SEED/Preprocessed_EEG/subject_15_training',
                     '../SEED/Preprocessed_EEG/subject_15_test']]

   iter = 1
   shuffle = True

   for subject in subject_files:
      print('Loading ' + str(subject))

      x_train, x_test, y_train, y_test = EEG_classifiers.load_data_two_files(
         training_path=subject[0] + '/train_matrix.csv', 
         test_path=subject[1] + '/test_matrix.csv')

      mlp_acc, mlp_mcc, mlp_auc = EEG_classifiers.run_mlp(
         "MLP_Classifier_SEED_"+timestamp,
         x_train, x_test, y_train, y_test)

      svm_acc, svm_mcc, svm_auc = EEG_classifiers.run_svm(
         "SVM_Classifier_SEED_"+timestamp,
         x_train, x_test, y_train, y_test)

      knn_acc, knn_mcc, knn_auc = EEG_classifiers.run_knn(
         "KNN_Classifier_SEED_"+timestamp,
         x_train, x_test, y_train, y_test)

      randf_acc, randf_mcc, randf_auc = EEG_classifiers.run_random_forest(
         "Random_Forest_Classifier_SEED_"+timestamp,
         x_train, x_test, y_train, y_test)

      adab_acc, adab_mcc, adab_auc = EEG_classifiers.run_ada_boost(
         "Ada_Boost_Classifier_SEED_"+timestamp,
         x_train, x_test, y_train, y_test)

      # CSV table for single run
      ##########################
      with open(subject[1]+'/results_'+timestamp+'.csv', 'w') as f:
         f.write('Classifier,acc,mcc,auc\n')
         f.write('SVM,' + str(svm_acc) + ',' + str(svm_mcc) + ',' + str(svm_auc) + '\n')
         f.write('KNN,' + str(knn_acc) + ',' + str(knn_mcc) + ',' + str(knn_auc) + '\n')
         f.write('Random Forest,' + str(randf_acc) + ',' + str(randf_mcc) + ',' + str(randf_auc) + '\n')
         f.write('ADA B,' + str(adab_acc) + ',' + str(adab_mcc) + ',' + str(adab_auc) + '\n')
         f.write('MLP,' + str(mlp_acc) + ',' + str(mlp_mcc) + ',' + str(mlp_auc) + '\n')

      # CSV table for multi run
      #########################
      if os.path.isfile('results_multi_'+timestamp+'.csv'):
         with open('results_multi_'+timestamp+'.csv', 'a') as f:
            f.write(str(svm_acc) + ',' + str(svm_mcc) + ',' + str(svm_auc) + ','
               + str(knn_acc) + ',' + str(knn_mcc) + ',' + str(knn_auc) + ','
               + str(randf_acc) + ',' + str(randf_mcc) + ',' + str(randf_auc) + ','
               + str(adab_acc) + ',' + str(adab_mcc) + ',' + str(adab_auc) + ','
               + str(mlp_acc) + ',' + str(mlp_mcc) + ',' + str(mlp_auc) + '\n')
      else:
         with open('results_multi_'+timestamp+'.csv', 'w') as f:
            f.write('svm_acc, svm_mcc, svm_auc, '
               + 'knn_acc, knn_mcc, knn_auc, '
               + 'randf_acc, randf_mcc, '
               + 'randf_auc, adab_acc, adab_mcc, '
               + 'adab_auc, mlp_acc, mlp_mcc, mlp_auc\n')
            f.write(str(svm_acc) + ',' + str(svm_mcc) + ',' + str(svm_auc) + ','
               + str(knn_acc) + ',' + str(knn_mcc) + ',' + str(knn_auc) + ','
               + str(randf_acc) + ',' + str(randf_mcc) + ',' + str(randf_auc) + ','
               + str(adab_acc) + ',' + str(adab_mcc) + ',' + str(adab_auc) + ','
               + str(mlp_acc) + ',' + str(mlp_mcc) + ',' + str(mlp_auc) + '\n')

   print('Done run_all_classifiers_once_all_subjects')
