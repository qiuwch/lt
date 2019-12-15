# Load model and images and infer predictions.
# Try to mimic python3 train.py   --eval --eval_dataset val   --config  experiments/human36m/eval/human36m_alg.yaml   --logdir ./logs
# Avoid using data loader for my purpose.
from tqdm import tqdm
from easydict import EasyDict as edict
import torch
import numpy as np
from torch.utils.data import DataLoader

from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet

from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.datasets import human36m
from mvn.datasets import utils as dataset_utils

from mvn.utils.vis import draw_2d_pose, fig_to_array
from mvn.utils.img import image_batch_to_numpy, to_numpy, denormalize_image, resize_image
from mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion

import matplotlib
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
matplotlib.use('Agg')

import pdb
import mocap.data
import mocap.model
import mocap.vis

config_path = 'experiments/human36m/eval/human36m_alg.yaml'

# def load_model(): return load_model_alg()

def test_data_reader():
    data_reader = mocap.data.H36mDataReader()
    cameras = data_reader.load_cameras()
    images = data_reader.load_images()

def main():
    model = mocap.model.AlgModel()
    data_reader = mocap.data.H36mDataReader()
    vis = mocap.vis.Visualizer()
    for iter_i in tqdm(range(100)):
        cameras = data_reader.load_cameras(iter_i)
        images = data_reader.load_images(iter_i)
        prediction = model.infer(cameras, images)
        for batch_index in range(images.shape[0]):
            fig = vis.vis_batch_2d(images, prediction.d2, cameras, prediction.d3, batch_index)
            plt.imsave('vis_%d_%d.png' % (iter_i, batch_index), fig)
    pdb.set_trace()

if __name__ == '__main__':
    # test_data_reader()
    main()

# TODO, try inference on NTU dataset.
# Take 2D keypoints from other dataset and combine the triangulation code.