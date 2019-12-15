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

config_path = 'experiments/human36m/eval/human36m_alg.yaml'

class AlgModel:
    def __init__(self, config=None):
        if not config:
            config = cfg.load_config(config_path)
        self.config = config
        self.device = torch.device(0)
        self.load()

    def load(self):
        device = self.device
        config = self.config

        model = AlgebraicTriangulationNet(config, device=device).to(device)
        state_dict = torch.load(config.model.checkpoint)

        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model

    def infer(self, cameras, images):
        keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred \
            = self.model(images, cameras, None)
        prediction = edict()
        prediction.d3 = keypoints_3d_pred
        prediction.d2 = keypoints_2d_pred
        return prediction

def load_model():
    return load_model_alg()


class Visualizer:
    # Visualize results
    def vis_batch_2d(self, images_batch, keypoints_2d_batch, proj_matricies_batch, keypoints_3d_batch_pred, batch_index):
        # n_views, n_joints = heatmaps_batch.shape[1], heatmaps_batch.shape[2]
        n_views, n_joints = keypoints_2d_batch.shape[1], keypoints_2d_batch.shape[2]
        # TODO, check it.

        # config
        # batch_index = 0 # TODO, check this?
        max_n_cols = 10
        size = 5
        kind = 'human36m'
        pred_kind = kind

        n_rows = 3
        n_cols = min(n_views, max_n_cols)
        
        fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * size, n_rows * size))
        axes = axes.reshape(n_rows, n_cols)

        row_i = 0
        # images
        axes[row_i, 0].set_ylabel("image", size='large')

        images = image_batch_to_numpy(images_batch[batch_index])
        images = denormalize_image(images).astype(np.uint8)
        images = images[..., ::-1]  # bgr -> rgb

        for view_i in range(n_cols):
            axes[row_i][view_i].imshow(images[view_i])
        row_i += 1

        # 2D keypoints (pred)
        if keypoints_2d_batch is not None:
            axes[row_i, 0].set_ylabel("2d keypoints (pred)", size='large')

            keypoints_2d = to_numpy(keypoints_2d_batch)[batch_index]
            for view_i in range(n_cols):
                axes[row_i][view_i].imshow(images[view_i])
                draw_2d_pose(keypoints_2d[view_i], axes[row_i][view_i], kind=kind)
            row_i += 1

        # 2D keypoints (pred projected)
        axes[row_i, 0].set_ylabel("2d keypoints (pred projected)", size='large')

        for view_i in range(n_cols):
            axes[row_i][view_i].imshow(images[view_i])
            keypoints_2d_pred_proj = project_3d_points_to_image_plane_without_distortion(proj_matricies_batch[batch_index, view_i].detach().cpu().numpy(), keypoints_3d_batch_pred[batch_index].detach().cpu().numpy())
            draw_2d_pose(keypoints_2d_pred_proj, axes[row_i][view_i], kind=pred_kind)
        row_i += 1

        fig.tight_layout()
        fig_image = fig_to_array(fig)
        plt.close('all')
        return fig_image

    def vis_batch_3d(self, keypoints_3d_batch_pred):
        pass

def test_data_reader():
    data_reader = H36mDataReader()
    cameras = data_reader.load_cameras()
    images = data_reader.load_images()

def main():
    model = AlgModel()
    data_reader = H36mDataReader()
    vis = Visualizer()
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