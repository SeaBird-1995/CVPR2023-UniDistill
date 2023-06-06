import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, num_bev_features, collapse_z=True):
        super().__init__()
        self.num_bev_features = num_bev_features
        self.collapse_z = collapse_z

    def forward(self, encoded_spconv_tensor, encoded_spconv_tensor_stride):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        spatial_features = encoded_spconv_tensor.dense()
        if self.collapse_z:
            N, C, D, H, W = spatial_features.shape
            spatial_features = spatial_features.view(N, C * D, H, W)
        return spatial_features, encoded_spconv_tensor_stride
