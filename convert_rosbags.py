from stream_rosbags import data_generator

import numpy as np
from tqdm import tqdm
from rospy_message_converter import message_converter
from label_msgs.msg import ImageLabels

import argparse
import pathlib
import glob
import json
import os

def cli():

    parser = argparse.ArgumentParser()
    parser.add_argument('rosbags',
        type=pathlib.Path,
        help='Path to your data directory which contains rosbags'
    )
    parser.add_argument('output_path',
        type=pathlib.Path,
        help='Path where your rosbags will be extracted'
    )
    parser.add_argument('--limit_by_N', type=int)
    parser.add_argument('--source_robot_name', type=str, default='small_scout_1')
    parser.add_argument('--skip_messages_without_labels', action='store_true', default=False)

    args = parser.parse_args()

    gen = tqdm(data_generator(
        args.rosbags,
        limit_by_N=args.limit_by_N,
        source_robot_name=args.source_robot_name,
        skip_messages_without_labels=args.skip_messages_without_labels,
        batches=None,
        yield_filename=True,
    ))

    n, prev_filename = None, None
    for filename, image, label in gen:

        if filename != prev_filename:
            prev_filename = filename
            n = 0

        output_dir = args.output_path / (pathlib
            .Path(filename)
            .relative_to(args.rosbags)
            .with_suffix(''))
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_file = output_dir / pathlib.Path(f'{n:06}.npy')
        label_file = output_dir / pathlib.Path(f'{n:06}.json')

        img = (np
            .frombuffer(image.data, dtype=np.dtype('B'))
            .reshape((image.height, image.width, 3))
        )
        with open(image_file, 'wb') as f:
            np.save(f, img)

        def repack(obj, name, dims):
            obj[name] = np.array(obj[name]).reshape(dims).tolist()

        with open(label_file, 'w') as f:
            obj = message_converter.convert_ros_message_to_dictionary(label)
            repack(obj, 'P', (3,4))
            for ln, ll in enumerate(obj['object_labels']):
                repack(ll, 'object__to__base_footprint', (4,4))
                repack(ll, 'base_footprint__to__camera', (4,4))
                repack(ll, 'object__to__camera', (4,4))

            json.dump(obj, f, indent=2)

        n += 1

if __name__ == '__main__':
    cli()
