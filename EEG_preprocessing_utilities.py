"""
A quick tutorial.

## Version history:
May 2020:
   Original script by Solange Cerny [scerny], Aston University
"""

import dask.dataframe as dd
import dask
import csv
import time
import sys


def remove_redundancies_shuffle(feature_matrix_file, to_rm):

    print (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),'Start')

    with open(feature_matrix_file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # gets the first line

    removed_columns = []
    keeping_columns = []

    for column in header:
        #print(column)

        remove = False

        for rm in to_rm:
            if rm in column:
                remove = True
                removed_columns.append(column) # 558 (23065)
                break
    
        if not remove:
            keeping_columns.append(column)   # 22507 (23065)

    print (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),'done columns')


    #df = dd.read_csv(feature_matrix_file, sample = 2000000)
    #df = dd.read_csv(feature_matrix_file, sample = 2000000, usecols=['lag1_mean_1', 'lag1_mean_3', 'lag1_mean_5'])
    df = dd.read_csv(feature_matrix_file, sample = 2000000, usecols=keeping_columns)

    #dask_head = df.head()
    #dask_columns = dask_head.columns
    #print(dask_columns)

    result_file = feature_matrix_file.replace('.csv', '_' + str(int(time.time())) + '.csv')
    #futs = df.to_csv('output/aaaa-*.csv', compute=False, index=False)
    #futs = df.to_csv('output/' + result_file, compute=False, index=False, single_file=True)
    futs = df.to_csv(result_file, compute=False, index=False, single_file=True)
    _, l = dask.compute(futs, df.size)

    print (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),'Done')

    #df = dd.read_csv('DASK_TEST.csv', sample = 2000000, usecols=['get_time_slice', 'calc_feature_vector', 'aaa'])
    #futs = df.to_csv('output/aaaa-*.csv', compute=False, index=False)
    #_, l = dask.compute(futs, df.size)



if __name__ == '__main__':
	"""
	Main function. The parameters for the script are the following:
		[1] feature_matrix_file: ...
		[2] to_rm: ...
	
	Author:
		Original by [scerny]
    """

	if len(sys.argv) != 3:
		print ('arg1: feature_matrix_file\narg2: to_rm')
		sys.exit(-1)
	feature_matrix_file = sys.argv[1]
	to_rm = sys.argv[2]
	Remove_redundancies_shuffle(feature_matrix_file, to_rm)
