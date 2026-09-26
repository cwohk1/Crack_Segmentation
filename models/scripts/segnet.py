"""Compact custom CNN with indexed pooling; not the full VGG16 SegNet."""
import torch
from torch import nn
from torch.nn import functional as F


class SegNet(nn.Module):
    def __init__(self, channels=3, num_class=1, init_f=16):
        super().__init__()
        # Keep parameter names/order so existing weights and optimizers still load.
        self.conv1 = nn.Conv2d(channels, init_f, 3, padding=1)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, 3, padding=1)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, 3, padding=1)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, 3, padding=1)
        self.conv5 = nn.Conv2d(8 * init_f, 16 * init_f, 3, padding=1)

        self.conv_up1 = nn.Conv2d(16 * init_f, 8 * init_f, 3, padding=1)
        self.conv_up2 = nn.Conv2d(8 * init_f, 4 * init_f, 3, padding=1)
        self.conv_up3 = nn.Conv2d(4 * init_f, 2 * init_f, 3, padding=1)
        self.conv_up4 = nn.Conv2d(2 * init_f, init_f, 3, padding=1)
        self.conv_out = nn.Conv2d(init_f, num_class, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        # Old checkpoints lack this marker and retain their bilinear behavior.
        self.register_buffer('_unpool_version', torch.tensor(1, dtype=torch.int64))
        self.indexed_unpool = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + '_unpool_version'
        if key not in state_dict:
            state_dict[key] = self._unpool_version.new_tensor(0)
        version = int(state_dict[key].item())
        if version not in (0, 1):
            error_msgs.append(f'Unsupported SegNet unpool version: {version}')
        self.indexed_unpool = version == 1
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        indices, sizes = [], []
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            x = F.relu(conv(x))
            sizes.append(x.size())
            x, index = self.pool(x)
            indices.append(index)
        x = F.relu(self.conv5(x))

        for conv, index, size in zip(
            (self.conv_up1, self.conv_up2, self.conv_up3, self.conv_up4),
            reversed(indices), reversed(sizes),
        ):
            if self.indexed_unpool:
                # Match channels to the corresponding encoder's pooling indices.
                x = F.relu(conv(x))
                x = self.unpool(x, index, output_size=size)
            else:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
                x = F.relu(conv(x))
        return self.conv_out(x)
