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
