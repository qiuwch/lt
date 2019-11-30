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

# Extend the data reader later to support using folder and videos.
# Only h36m supports training.
class BaseDataReader:
    def load_images(self):
        pass

    def load_cameras(self):
        pass

    def __len__(self):
        return 0

# Try this for H36M -> NTU -> Real-world webcams
class FolderDataReader:
    # Read test images from a folder
    def __init__(self, folder_paths, calibration_result):
        '''
        folder_paths: A list of sync videos for inference
        calibration_result: camera extrinsics for triangulation
        '''
        pass

class VideoDataReader:
    def __init__(self, video_paths, calibration_result):
        '''
        video_paths: A list of sync videos for inference
        calibration_result: camera extrinsics for triangulation
        '''
        pass

class CameraDataReader(BaseDataReader):
    def __init__(self, camera_ids, calibration_result):
        ''' 
        camera_ids: opencv usb id
        calibration_result: camera extrinsics for triangulation
        ''' 
        pass

class H36mDataReader(BaseDataReader):
    def __init__(self, config=None):
        if not config:
            config = cfg.load_config(config_path)
        self.config = config
        self.device = torch.device(0)
        self.setup_dataloader()

    def setup_dataloader(self):
        config = self.config

        val_dataset = human36m.Human36MMultiViewDataset(
            h36m_root=config.dataset.val.h36m_root,
            pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None,
            train=False,
            test=True,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
            labels_path=config.dataset.val.labels_path,
            with_damaged_actions=config.dataset.val.with_damaged_actions,
            retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
            scale_bbox=config.dataset.val.scale_bbox,
            kind=config.kind,
            undistort_images=config.dataset.val.undistort_images,
            ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
            crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
            shuffle=config.dataset.val.shuffle,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                    min_n_views=config.dataset.val.min_n_views,
                                                    max_n_views=config.dataset.val.max_n_views),
            num_workers=config.dataset.val.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn
        )
        # Use dataloader to iterate over data
        self.dataloader = val_dataloader
        self.batches = []

    # How to define image format and camera format?
    def load_images(self, iter_i = 0):
        images_batch, proj_matricies_batch = self.load_batch(iter_i) 
        return images_batch

    def load_cameras(self, iter_i = 0):
        images_batch, proj_matricies_batch = self.load_batch(iter_i) 
        return proj_matricies_batch
    
    def load_batch(self, iter_i = 0):
        # iterator = enumerate(self.dataloader)
        # batch = iterator[iter_i]
        while len(self.batches) <= iter_i:
            _batch = next(iter(self.dataloader)) # TODO: Avoid using this.
            self.batches.append(_batch)
        batch = self.batches[iter_i]
        images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch \
            = dataset_utils.prepare_batch(batch, self.device, self.config)
        return images_batch, proj_matricies_batch

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