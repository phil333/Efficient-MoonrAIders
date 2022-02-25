#!/usr/bin/env python3

import os
import sys
import time
import numpy as np


# variables
######################################
input_database_name = sys.argv[1]
#input_database_name = 'small_scout_1_record_traverse.db'
directory = os.path.dirname(input_database_name)
print('directory       '+ directory)
file = os.path.basename(input_database_name)
filename = os.path.splitext(file)[0]
print('filename:       '+ filename)
result_text = '' # print all results at the bottom
counter = 0
print("input database: " + input_database_name)

# function that retrieves two values from
######################################
def get_values(input,database_name):

    out2 = [line for line in input.split('\n') if not line.startswith("\x1b[33m[ WARN]")]

    out3 = out2[1].split('max=')
    out4 = out3[1].split('m,')
    out_max = float(out4[0].replace(',','.'))

    out5 = out2[1].split('odom=')
    out6 = out5[1].split('m) ')[0]
    out_odom = float(out6.replace(',','.')  )

    out7 = out2[1].split('lin=')
    out8 = out7[1].split('m ')[0]
    out_lin = float(out8.replace(',','.')  )

    orig_rsme = get_rsme_value(database_name)

    print('get_values: lin: ' + str(out_lin) + ' max: ' + str(out_max) + ' odom: ' + str(out_odom) + ' rsme: ' + str(orig_rsme))

    return [out_lin, out_max, out_odom, orig_rsme]


# function that retrieves RSME from DB (database needs to be reprocessed)
######################################
def get_rsme_value(database_name):
    os.popen('rtabmap-report Gt/Translational_rmse/m --export ' + database_name).read()

    results_file = 'Stat-' + database_name.replace('/','-') + '.txt'

    with open(results_file, "r") as file:
        first_line = file.readline()
        for last_line in file:
            pass

    out = last_line.split('\t')

    return out[1]


# print out rtabmap report or original databse
###############################################
out = os.popen('rtabmap-report ' + input_database_name).read()
print(out)
orig_val = get_values(out,input_database_name)


# open csv file that we print results into
###############################################
csv_file = directory + '/results.csv'
does_results_file_exist = os.path.exists(csv_file)

results_csv_file = open(csv_file, "a")
if(not does_results_file_exist):
    results_csv_file.write('database name, orig lin(m), orig max(m), orig odom(m), orig rsme(m), updated lin(m), updated max(m), updated odom(m), updated rsme(m), command used\n')



# define tests that will be run
##############################################
test_list = [] # list that contains the tests we aim to run

grid_logspace = 10 ** np.linspace(-3.999, 5, 8)
skip_optimisation = 0.0001

# DONE
### testing unit matrix

test_command = 'python3 rtabmap-db-covariance-modifier.py CONSTANT_ROTATION_AND_POSITION'
test_command += f' --pos_constant_value 1'
test_command += f' --rot_constant_value 1'
test_list.append(test_command)

### testing const matrix covariances

for pos_constant_value in grid_logspace:
    # single test instructions
    #################################################
    test_command = 'python3 rtabmap-db-covariance-modifier.py CONSTANT_ROTATION_AND_POSITION'
    test_command += f' --pos_constant_value {pos_constant_value}'
    test_command += f' --rot_constant_value {skip_optimisation}'
    test_list.append(test_command)

for rot_constant_value in grid_logspace:
    # single test instructions
    #################################################
    test_command = 'python3 rtabmap-db-covariance-modifier.py CONSTANT_ROTATION_AND_POSITION'
    test_command += f' --pos_constant_value {skip_optimisation}'
    test_command += f' --rot_constant_value {rot_constant_value}'
    test_list.append(test_command)


for pos_constant_value in grid_logspace:
    for rot_constant_value in grid_logspace:
        # single test instructions
        #################################################
        test_command = 'python3 rtabmap-db-covariance-modifier.py CONSTANT_ROTATION_AND_POSITION'
        test_command += f' --pos_constant_value {pos_constant_value}'
        test_command += f' --rot_constant_value {rot_constant_value}'
        test_list.append(test_command)

### testing variable covariance depending on pred dist to object
pos_mapped_min_max = [
    (1e7,  1e6),
    (1e6,  1e5),
    (1e5,  1e4),
    (1e4,  1e3),
    (1e3,  1e2),
    (1e2,  1e1),
    (1e1,  1e0),
]

for min_distance in [1, 10, 20, 30]:
    for delta_distance in [10, 30]:
        for (pos_mapped_value_min, pos_mapped_value_max) in pos_mapped_min_max:
            for y_multiplier in [1, 100, 1000]:
                for z_multiplier in [1, 0.01, 0.001]:
                    # single test instructions
                    test_command  = 'python3 rtabmap-db-covariance-modifier.py ROT_AND_POS_ON_DISTANCE_TO_OBJECT'
                    test_command += ' --distance_to_object_min_threshold ' + str(min_distance)
                    test_command += ' --distance_to_object_max_threshold ' + str(min_distance + delta_distance)
                    test_command += ' --pos_mapped_value_min ' + str(pos_mapped_value_min)
                    test_command += ' --pos_mapped_value_max ' + str(pos_mapped_value_max)
                    test_command += ' --pos_mapped_value_under_min_distance_threshold ' + str(pos_mapped_value_min)
                    test_command += ' --pos_mapped_value_above_max_distance_threshold ' + str(pos_mapped_value_max)
                    test_command += ' --rot_mapped_value_min 0.0001'
                    test_command += ' --rot_mapped_value_max 0.0001'
                    test_command += ' --rot_mapped_value_under_min_distance_threshold 0.0001'
                    test_command += ' --rot_mapped_value_above_max_distance_threshold 0.0001'
                    test_command += f' --pos_y_axis_multiplier {y_multiplier}'
                    test_command += f' --pos_z_axis_multiplier {z_multiplier}'
                    test_list.append(test_command)


### best setting
test_command  = 'python3 rtabmap-db-covariance-modifier.py ROT_AND_POS_ON_DISTANCE_TO_OBJECT'
test_command += ' --distance_to_object_min_threshold 1'
test_command += ' --distance_to_object_max_threshold 31'
test_command += ' --pos_mapped_value_min 10000'
test_command += ' --pos_mapped_value_max 100'
test_command += ' --pos_mapped_value_under_min_distance_threshold 10000'
test_command += ' --pos_mapped_value_above_max_distance_threshold 100'
test_command += ' --rot_mapped_value_min 0.0001'
test_command += ' --rot_mapped_value_max 0.0001'
test_command += ' --rot_mapped_value_under_min_distance_threshold 0.0001'
test_command += ' --rot_mapped_value_above_max_distance_threshold 0.0001'
test_command += f' --pos_y_axis_multiplier 100'
test_command += f' --pos_z_axis_multiplier 0.01'
test_command += f' --odometry_multiplier 2'
test_list.append(test_command)

print('total number of tests: ' + str(len(test_list)) )

# # loop that runs tests
# ################################################
for test_command in test_list:
    print('################################# test ' + str(counter))
    print(test_command)
    start_time = time.time()

    modified_covarience__db = directory + '/' + filename + '_modi_covari.db'
    modified_reprocessed_db = directory + '/' + filename + '_reprocessed.db'

    # modify database covariences
    os.system(test_command + ' ' + input_database_name + ' ' + modified_covarience__db )
    print('# done database modifications')

    # reprocess graph optimisation of database
    os.popen('rtabmap-reprocess ' + modified_covarience__db + ' ' + modified_reprocessed_db).read()

    print('# done reprocessing')

    # print out rtabmap report
    out = os.popen('rtabmap-report ' + modified_reprocessed_db).read()

    print(out)
    print('# done report')


    val = get_values(out, modified_reprocessed_db)
    print(val)

    print_text = modified_reprocessed_db + ', ' + str(orig_val[0]) + ', ' + str(orig_val[1]) + ', ' + str(orig_val[2]) + ', ' + str(orig_val[3])  + ', '  + str(val[0]) + ', ' + str(val[1]) + ', ' + str(val[2]) + ', ' + str(val[3]) + ', '  + test_command[41:] + '\n'
    result_text +=  modified_reprocessed_db + ', original: '  + str(orig_val[0]) + ', ' + str(orig_val[1]) + ', ' + str(orig_val[2]) + ', ' + str(orig_val[3]) + ' updated: ' + str(val[0]) + ', ' + str(val[1]) + ', ' + str(val[2]) + ', ' + str(val[3]) + '\n'

    # write results to CSV file
    results_csv_file.write(print_text)

    difference = time.time() - start_time
    print('############ seconds: ' +str(difference))

    counter += 1


results_csv_file.close()
print(result_text)
