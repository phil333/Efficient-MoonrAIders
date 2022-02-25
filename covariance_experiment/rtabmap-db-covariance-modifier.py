import sqlite3
import argparse
from struct import unpack, pack
import pandas as pd
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R

from shutil import copyfile
from pathlib import Path

LINK_TYPE_ENUM = {
    'Neighbor': 0,
    'GlobalClosure': 1,
    'LocalSpaceClosure': 2,
    'LocalTimeClosure': 3,
    'UserClosure': 4,
    'VirtualClosure': 5,
    'NeighborMerged': 6,
    'PosePrior': 7,
    'Landmark': 8
}

MODIFICATION_TYPES = {
    'CONSTANT_ROTATION_ONLY',
    'CONSTANT_POSITION_ONLY',
    'CONSTANT_ROTATION_AND_POSITION',
    'ROT_ON_DISTANCE_TO_OBJECT',
    'POS_ON_DISTANCE_TO_OBJECT',
    'ROT_AND_POS_ON_DISTANCE_TO_OBJECT'
}

def unpack_data(series, dim, type='f'):
    size = type * dim[0] * dim[1]
    return series.map(lambda x: np.array(unpack(size, x)).reshape(dim))

def pack_data(series, dim, type='f'):
    size = type * dim[0] * dim[1]
    return series.map(lambda x: pack(size, *x.reshape(-1)))

def update_link(conn, from_id, information_matrix):
    """
    update priority, begin_date, and end date of a task
    :param conn:
    :param task:
    :return: project id
    """
    SQL = ''' UPDATE Link
              SET information_matrix = ?
              WHERE from_id = ?'''
    cur = conn.cursor()
    cur.execute(SQL, (information_matrix, from_id))
    conn.commit()

def main():

    parser = argparse.ArgumentParser(description='Modify RTabMap\'s covariance matrices')

    parser.add_argument('modification_type', type=str, choices=MODIFICATION_TYPES, default='CONSTANT_ROTATION_ONLY',
                        help='Type of modification to the covariance matrices')

    parser.add_argument('input_db_filepath', type=Path,
                        help='Path to an input RTabMap database')
    parser.add_argument('output_db_filepath', type=Path,
                        help='Path to an output file')

    parser.add_argument('-lt', '--link_type', type=str, default='Landmark',
                        help='Link type to modify the covariances')

    parser.add_argument('--rot_constant_value', type=float, default=9999.0,
                        help='Constant value to replace all covariance matrices (rotation part)')

    parser.add_argument('--pos_constant_value', type=float, default=9999.0,
                        help='Constant value to replace all covariance matrices (position part)')


    parser.add_argument('--rot_mapped_value_min', type=float, default=1.0,
                        help='Minimum rotation covariance value, when mapping rotation based on distance to object.')

    parser.add_argument('--pos_y_axis_multiplier', type=float, default=1.0,
                        help='multiplier used to reduece or increase the importance the y axis position')

    parser.add_argument('--pos_z_axis_multiplier', type=float, default=1.0,
                        help='multiplier used to reduece or increase the importance the y axis position')


    parser.add_argument('--rot_mapped_value_max', type=float, default=9999.0,
                        help='Maximum rotation covariance value, when mapping rotation based on distance to object.')

    parser.add_argument('--rot_mapped_value_under_min_distance_threshold', type=float, default=9999.0,
                        help='Rotation covariance value, when distance to object is under min threshold')

    parser.add_argument('--rot_mapped_value_above_max_distance_threshold', type=float, default=9999.0,
                        help='Rotation covariance value, when distance to object is above max threshold')


    parser.add_argument('--pos_mapped_value_min', type=float, default=1.0,
                        help='Minimum position covariance value, when mapping rotation based on distance to object.')

    parser.add_argument('--pos_mapped_value_max', type=float, default=9999.0,
                        help='Maximum position covariance value, when mapping rotation based on distance to object.')

    parser.add_argument('--pos_mapped_value_under_min_distance_threshold', type=float, default=9999.0,
                        help='Position covariance value, when distance to object is under min threshold')

    parser.add_argument('--pos_mapped_value_above_max_distance_threshold', type=float, default=9999.0,
                        help='Position covariance value, when distance to object is above max threshold')


    parser.add_argument('--distance_to_object_min_threshold', type=float, default=5.0,
                        help='Minimum distance to object threshold, when mapping distance to object to covariance values')

    parser.add_argument('--distance_to_object_max_threshold', type=float, default=40.0,
                        help='Maximum distance to object threshold, when mapping distance to object to covariance values')

    parser.add_argument('--odometry_multiplier', type=float,
                        help='Multiply odometry covariances')


    args = parser.parse_args()

    try:
        copyfile(args.input_db_filepath, args.output_db_filepath)
    except Exception as e:
        print('Error while cloing the database. Make sure paths are correct and you have space on the disk!')

    con = sqlite3.connect(str(args.output_db_filepath))

    SQL = '''
        SELECT l.*, n.*
        FROM link l
        INNER JOIN node n on n.id = l.from_id
        WHERE l."type" = 8
    '''
    df = pd.read_sql(SQL, con)

    if not args.odometry_multiplier is None:
        SQL = '''
            SELECT l.*, n.*
            FROM link l
            INNER JOIN node n on n.id = l.from_id
            WHERE l."type" = 0
        '''
        odometry_df = pd.read_sql(SQL, con)
        odom_information_matrix_unpacked = unpack_data(odometry_df['information_matrix'], (6, 6), 'd')
        odom_information_matrix_changed = odom_information_matrix_unpacked.apply(lambda x: x * args.odometry_multiplier)

        odometry_df['information_matrix'] = pack_data(odom_information_matrix_changed, (6, 6), 'd')
        for _, row in odometry_df.iterrows():
            update_link(con, row.from_id, row.information_matrix)

    information_matrix_unpacked = unpack_data(df['information_matrix'], (6, 6), 'd')
    transform_unpacked = unpack_data(df['transform'], (3, 4), 'f')

    distances_to_object = (transform_unpacked
        .map(lambda x: (x[:2, 3]**2).sum() ** 0.5) # distance in xy plane
    )

    if args.modification_type == 'CONSTANT_ROTATION_ONLY':
        information_matrix_changed = (information_matrix_unpacked
        #     .map(lambda x: np.eye(6)
        #         * [0, 0, 0, 1, 1, 1]
        #         * args.rot_constant_value
        #     )
        # )
            .map(lambda x: np.eye(6) * (
                  (np.array([1, args.pos_y_axis_multiplier, args.pos_z_axis_multiplier, 0, 0, 0]) * 9999)
                + (np.array([0, 0, 0, 1, 1, 1]) * args.rot_constant_value)
                )
            )
        )
    elif args.modification_type == 'CONSTANT_POSITION_ONLY':
        information_matrix_changed = (information_matrix_unpacked
        #     .map(lambda x: np.eye(6)
        #         * [1, 1, 1, 0, 0, 0]
        #         * args.pos_constant_value
        #     )
        # )
            .map(lambda x: np.eye(6) * (
                  (np.array([1, args.pos_y_axis_multiplier, args.pos_z_axis_multiplier, 0, 0, 0]) * args.pos_constant_value)
                + (np.array([0, 0, 0, 1, 1, 1]) * 9999)
                )
            )
        )
    elif args.modification_type == 'CONSTANT_ROTATION_AND_POSITION':
        information_matrix_changed = (information_matrix_unpacked
            .map(lambda x: np.eye(6) * (
                  (np.array([1, args.pos_y_axis_multiplier, args.pos_z_axis_multiplier, 0, 0, 0]) * args.pos_constant_value)
                + (np.array([0, 0, 0, 1, 1, 1]) * args.rot_constant_value)
                )
            )
        )
    elif args.modification_type == 'ROT_ON_DISTANCE_TO_OBJECT':

        mapped_dists_to_rot_cov = np.interp(
            distances_to_object,
            xp=[
                args.distance_to_object_min_threshold,
                args.distance_to_object_max_threshold
            ],
            fp=[args.rot_mapped_value_min, args.rot_mapped_value_max],
            left=args.rot_mapped_value_under_min_distance_threshold,
            right=args.rot_mapped_value_above_max_distance_threshold
        ).reshape((-1, 1)) * [0,0,0,1,1,1]

        information_matrix_changed = pd.Series(
            list(mapped_dists_to_rot_cov[:, np.newaxis] * np.eye(6)),
            index=distances_to_object.index
        )

    elif args.modification_type == 'POS_ON_DISTANCE_TO_OBJECT':

        mapped_dists_to_pos_cov = np.interp(
            distances_to_object,
            xp=[
                args.distance_to_object_min_threshold,
                args.distance_to_object_max_threshold
            ],
            fp=[args.pos_mapped_value_min, args.pos_mapped_value_max],
            left=args.pos_mapped_value_under_min_distance_threshold,
            right=args.pos_mapped_value_above_max_distance_threshold
        ).reshape((-1, 1)) * [[1,1,1,0,0,0]]
        information_matrix_changed = pd.Series(
            list(mapped_dists_to_pos_cov[:, np.newaxis] * np.eye(6)),
            index=distances_to_object.index
        )

    elif args.modification_type == 'ROT_AND_POS_ON_DISTANCE_TO_OBJECT':

        mapped_dists_to_pos_cov = np.interp(
            distances_to_object,
            xp=[
                args.distance_to_object_min_threshold,
                args.distance_to_object_max_threshold
            ],
            fp=[args.pos_mapped_value_min, args.pos_mapped_value_max],
            left=args.pos_mapped_value_under_min_distance_threshold,
            right=args.pos_mapped_value_above_max_distance_threshold
        ).reshape((-1, 1)) * [[1,args.pos_y_axis_multiplier,args.pos_z_axis_multiplier,0,0,0]]

        mapped_dists_to_rot_cov = np.interp(
            distances_to_object,
            xp=[
                args.distance_to_object_min_threshold,
                args.distance_to_object_max_threshold
            ],
            fp=[args.rot_mapped_value_min, args.rot_mapped_value_max],
            left=args.rot_mapped_value_under_min_distance_threshold,
            right=args.rot_mapped_value_above_max_distance_threshold
        ).reshape((-1, 1)) * [0,0,0,1,1,1]

        mapped_dists_to_cov = mapped_dists_to_pos_cov + mapped_dists_to_rot_cov

        information_matrix_changed = pd.Series(
            list(mapped_dists_to_cov[:, np.newaxis] * np.eye(6)),
            index=distances_to_object.index
        )

    else:
        raise ValueError(f'modification type has to be in { {*MODIFICATION_TYPES} }')


    df['information_matrix'] = pack_data(information_matrix_changed, (6, 6), 'd')
    print(information_matrix_changed[0])
    for _, row in df.iterrows():
        update_link(con, row.from_id, row.information_matrix)

    con.close()

if __name__ == '__main__':
    main()
