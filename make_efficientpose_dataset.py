import numpy as np
import trimesh
import pyrender
import argparse, pathlib
from pyrender.constants import RenderFlags
import matplotlib.pyplot as plt
from pathlib import Path
import json, cv2
import os
from shutil import copyfile
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
import yaml

from tqdm.auto import tqdm

MIN_NUM_PIXELS_THRESHOLD = 10

models = {
    'processing_plant': {
        'object_id': 1,
        'color': (255, 0, 0),
        'rotation': R.from_euler('z', -90, degrees=True).as_matrix(),
        'translation': [0, 0, 0.65],
        'model_filename': 'processing_plant.ply'
    },
    'repair_station': {
        'object_id': 2,
        'color': (0, 0, 255),
        'rotation': R.from_euler('z', 0.78399, degrees=True).as_matrix(),
        'translation': [0, 0, 3.1],
        'model_filename': 'repair_station.ply'
    },
    'small_hauler_1': {
        'object_id': 3,
        'color': (0, 255, 0),
        'rotation': R.from_euler('z', 90, degrees=True).as_matrix(),
        'translation': [0, 0, 0],
        'model_filename': 'rover_03_hauler.ply'
    },
    'small_excavator_1': {
        'object_id': 4,
        'color': (255, 255, 0),
        'rotation': R.from_euler('z', 90, degrees=True).as_matrix(),
        'translation': [0, 0, 0],
        'model_filename': 'rover_03_excavator.ply'
    }
    # 'small_scout_1': {
    #     'color': (255, 0, 0),
    #     'rotation': R.from_euler('z', 0, degrees=True).as_matrix(),
    #     'translation': [0, 0, 3.1],
    #     'model_filename': 'rover_03_scout.ply'
    # },
}

def draw_points(points, color, proj_mat, cv_image):
    img_plan_points = (proj_mat @ points)
    img_plan_points = (img_plan_points / img_plan_points[2])[:2].astype(int).T
    for point in img_plan_points:
        cv2.circle(cv_image, tuple(point), 0, color)
    for p1 in img_plan_points:
        for p2 in img_plan_points:
            if (p1 != p2).any():
                cv2.line(cv_image, tuple(p1), tuple(p2), color, thickness = 1)
    return img_plan_points

def create_linemod_model_dir(models, models_path, output_path):

    models_output_path = output_path / 'models'

    makedir(models_output_path)
    model_info = {}

    for name, model in models.items():
        obj_id = model['object_id']
        model_filename = model['model_filename']
        model_path = models_path / model_filename
        copyfile(model_path, models_output_path / f'obj_{obj_id:02d}.ply')

        plydata = PlyData.read(model_path)
        points = np.array(plydata['vertex'].data[['x', 'y', 'z']])
        points = np.array(points.tolist())
        p_min, p_max = points.min(axis=0), points.max(axis=0)
        size = p_max - p_min

        sample = points[np.random.randint(low=0, high=points.shape[0], size=100)]

        diameter = max([
            max([np.sqrt(((y-x)**2).sum()) for y in sample])
            for x in sample
        ])


        model_info[obj_id] = {
            'diameter': float(diameter),
            'min_x': float(p_min[0]),
            'min_y': float(p_min[1]),
            'min_z': float(p_min[2]),
            'size_x': float(size[0]),
            'size_y': float(size[1]),
            'size_z': float(size[2]),
        }

    with open(models_output_path / 'models_info.yml', 'w') as file:
        yaml.dump(model_info, file, default_flow_style=False)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_mesh_with_color(name, color):
    mesh = models[name]['model']
    mesh.visual.face_colors = [color] * mesh.faces.shape[0]
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


def deinterpolate_colors(image, models, label):
    color_mapping = {
        obj['object_name']: models[obj['object_name']]['color']
        for obj in label['object_labels'] if obj['object_name'] in models
    }

    # color mapping (object -> mask)
    colors = np.array(list(color_mapping.values())).T

    # de-interpolate stupid rendering
    closest = ((np.expand_dims(image, -1) - colors) ** 2).sum(axis=-2).argmin(axis=-1)
    closest = np.array([[colors.T[ele] for ele in row] for row in closest.tolist()], dtype=np.uint8)
    closest[(image == [0,0,0]).all(axis=2)] = 0

    return closest

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',
        type=pathlib.Path,
        help='Path to your extracted rosbags'
    )
    parser.add_argument('output_path',
        type=pathlib.Path,
        help='Path where your EfficientPose dataset will be created'
    )
    parser.add_argument('--skip_invalid_image', action='store_true', default=True)
    parser.add_argument('--class_name', type=str, default='processing_plant')
    parser.add_argument('--models_path', type=pathlib.Path, default=(Path.home() / 'Desktop/models'))

    args = parser.parse_args()

    output_path = args.output_path
    models_path = args.models_path
    input_path = args.input_path

    chosen_class_name = args.class_name
    skip_invalid_image = args.skip_invalid_image

    chosen_class_id = models[chosen_class_name]['object_id']

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_output_path = output_path / 'data/01'
    makedir(data_output_path)
    makedir(data_output_path / 'mask')
    makedir(data_output_path / 'rgb')

    for name, data in models.items():
        data['model'] = trimesh.load(models_path / data['model_filename'])
        mins = data['model'].vertices.min(axis=0)
        maxs = data['model'].vertices.max(axis=0)
        bbox_3d = np.array(np.meshgrid(*zip(mins, maxs))).T.reshape(-1, 3)
        data['bbox_3d'] = np.hstack([bbox_3d, np.ones((8, 1))])

    create_linemod_model_dir(models, models_path, output_path)
    #return

    linemod_gts = {}
    linemod_infos = {}
    linemod_train = []
    linemod_test = []

    for image_idx, image_npy in tqdm(enumerate(input_path.glob('*.npy'))):

        try:

            image_idx = f'{image_idx:06d}'

            label_file = image_npy.with_suffix('.json')
            with open(label_file, 'rt') as f:
                label = json.load(f)

            orig_image = np.load(image_npy)
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

            scene = pyrender.Scene(ambient_light=20000000, bg_color=(0,0,0))
            fx = 381.36246688113556
            fy = 381.36246688113556
            cx = 320.5
            cy = 240.5

            camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)

            cp = np.eye(4)
            cp[:3, :3] = R.from_euler('x', 180, degrees=True).as_matrix()
            scene.add(camera, pose=cp)

            class_transform = None

            if not len(label['object_labels']):
                continue

            invalid_image = False

            for obj in label['object_labels']:

                name = obj['object_name']
                model = models[name]

                bbox_3d = model['bbox_3d']

                rot, tran = model['rotation'], model['translation']
                color = model['color']

                mesh = get_mesh_with_color(name, color)

                object__to__camera = np.array(obj['object__to__camera'])

                t = np.eye(4)
                t[:3,:3] = rot
                t[:3, 3] = tran

                proj_mat = object__to__camera @ t

                if name == chosen_class_name:
                    class_transform = proj_mat

                ci = np.zeros((3, 4))
                ci[0,0] = fx
                ci[1,1] = fy
                ci[2,2] = 1
                ci[0,2] = cx
                ci[1,2] = cy

                bbox_3d_in_camera_frame = proj_mat @ bbox_3d.T

                bbox_3d_in_image_plane = ci @ bbox_3d_in_camera_frame

                bbox_3d_in_image_plane = (
                    bbox_3d_in_image_plane / bbox_3d_in_image_plane[2]
                )[:2].astype(int).T

                inside_image_plane  = (bbox_3d_in_image_plane[:, 0] >= 0)
                inside_image_plane &= (bbox_3d_in_image_plane[:, 0] <= label['width'])
                inside_image_plane &= (bbox_3d_in_image_plane[:, 1] >= 0)
                inside_image_plane &= (bbox_3d_in_image_plane[:, 1] <= label['height'])

                if (bbox_3d_in_camera_frame[2] < 0).any() or not inside_image_plane.all():
                    invalid_image = True

                scene.add(mesh, pose=proj_mat)

            if invalid_image and skip_invalid_image:
                continue

            r = pyrender.OffscreenRenderer(640, 480)
            image, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
            deinterpolate_colors(image, models, label)

            mask = image.copy()
            mask[(image != models[chosen_class_name]['color']).any(axis=2)] = 0
            mask[(image == models[chosen_class_name]['color']).all(axis=2)] = [255, 255, 255]

            num_pixels = (mask[:, :, 0] == 255).sum()

            if (num_pixels < MIN_NUM_PIXELS_THRESHOLD) and skip_invalid_image:
                continue

            mask_ys, mask_xs = np.where((mask == [255, 255, 255]).all(axis=2))

            if (len(mask_xs) > 0 and len(mask_ys) > 0):

                x_min, x_max = mask_xs.min(), mask_xs.max()
                y_min, y_max = mask_ys.min(), mask_ys.max()

                bbox = np.array([x_min, y_min, (x_max-x_min), (y_max-y_min)])

                cam_R_m2c = class_transform[:3, :3].reshape(-1).tolist()
                cam_t_m2c = class_transform[:3, 3].reshape(-1).tolist()
                obj_bb = bbox.reshape(-1).tolist()

            elif skip_invalid_image:
                continue

            else:

                cam_R_m2c = np.eye(3).reshape(-1).tolist()
                cam_t_m2c = np.zeros((3, 1)).reshape(-1).tolist()
                obj_bb = np.zeros((4, 1)).reshape(-1).tolist()


            linemod_gts[int(image_idx)] = [{
                'cam_R_m2c': cam_R_m2c,
                'cam_t_m2c': cam_t_m2c,
                'obj_bb': obj_bb,
                'obj_id': chosen_class_id
            }]

            linemod_infos[int(image_idx)] = {
                'cam_K': [fx, 0, cx, 0, fy, cy, 0, 0, 1],
                'depth_scale': 1.0
            }

            cv2.imwrite(str(data_output_path / 'mask' / f'{image_idx}.png'), mask)
            cv2.imwrite(str(data_output_path / 'rgb' / f'{image_idx}.png'), orig_image)

            t_type = image_npy.parent.parent.name
            if t_type == 'train':
                linemod_train.append(image_idx)
            elif t_type == 'val':
                linemod_test.append(image_idx)

        except Exception as e:
            import traceback
            print(e, image_npy)
            print(traceback.format_exc())

    with open(data_output_path / 'gt.yml', 'w') as file:
        yaml.dump(linemod_gts, file, default_flow_style=False)

    with open(data_output_path / 'info.yml', 'w') as file:
        yaml.dump(linemod_infos, file, default_flow_style=False)

    with open(data_output_path / 'train.txt', 'w') as file:
        file.write('\n'.join(linemod_train))
        file.write('\n')

    with open(data_output_path / 'test.txt', 'w') as file:
        file.write('\n'.join(linemod_test))
        file.write('\n')

if __name__ == '__main__':
    main()
