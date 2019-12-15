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
