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