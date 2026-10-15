import torch
import torch.nn as nn
import torch.nn.functional as F

from models.generator.attention import CloudAttentionMap
from models.generator.fusion import CFA, OSGC
from models.generator.gated_conv import GatedConv2d
from utils.model_utils import weights_init


class Generator(nn.Module):
    """Generator network for optical-SAR fusion and cloud removal."""

    def __init__(
        self,
        optical_channels=5,
        sar_channels=4,
        out_channels=3,
        init_weights=True
    ):
        super().__init__()

        self.freeze_ec_bn = False

        # -----------------------------
        # Optical encoder-decoder branch
        # -----------------------------
        self.optical_ec_1 = self._make_gconv(
            optical_channels, 64, kernel_size=7, stride=2, padding=3, norm="none"
        )
        self.optical_ec_2 = self._make_gconv(
            64, 128, kernel_size=5, stride=2, padding=2
        )
        self.optical_ec_3 = self._make_gconv(
            128, 256, kernel_size=5, stride=2, padding=2
        )
        self.optical_ec_4 = self._make_gconv(
            256, 512, kernel_size=3, stride=2, padding=1
        )

        self.optical_dc_4 = self._make_gconv(
            512 + 256, 256, kernel_size=3, stride=1, padding=1
        )
        self.optical_dc_3 = self._make_gconv(
            256 + 128, 128, kernel_size=3, stride=1, padding=1
        )
        self.optical_dc_2 = self._make_gconv(
            128 + 64, 64, kernel_size=3, stride=1, padding=1
        )
        self.optical_dc_1 = self._make_gconv(
            64 + sar_channels, 64, kernel_size=3, stride=1, padding=1
        )

        # -------------------------
        # SAR encoder-decoder branch
        # -------------------------
        self.sar_ec_1 = self._make_gconv(
            sar_channels, 64, kernel_size=7, stride=2, padding=3, norm="none"
        )
        self.sar_ec_2 = self._make_gconv(
            64, 128, kernel_size=5, stride=2, padding=2
        )
        self.sar_ec_3 = self._make_gconv(
            128, 256, kernel_size=5, stride=2, padding=2
        )
        self.sar_ec_4 = self._make_gconv(
            256, 512, kernel_size=3, stride=2, padding=1
        )

        self.sar_dc_4 = self._make_gconv(
            512 + 256, 256, kernel_size=3, stride=1, padding=1
        )
        self.sar_dc_3 = self._make_gconv(
            256 + 128, 128, kernel_size=3, stride=1, padding=1
        )
        self.sar_dc_2 = self._make_gconv(
            128 + 64, 64, kernel_size=3, stride=1, padding=1
        )
        self.sar_dc_1 = self._make_gconv(
            64 + out_channels + 2, 64, kernel_size=3, stride=1, padding=1
        )

        # -----------------------------
        # Optical-SAR global coupling
        # -----------------------------
        self.osgc_opt = OSGC(
            dim=64,
            input_resolution=(256, 256),
            num_heads=8,
            window_size=8,
            shift_size=4,
            is_SAR=False
        )
        self.osgc_sar = OSGC(
            dim=64,
            input_resolution=(256, 256),
            num_heads=8,
            window_size=8,
            shift_size=4,
            is_SAR=True
        )

        self.fusion_layer1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Feature similarity and multi-scale aggregation.
        self.ccfs_msfa = CFA(in_channels=64, out_channels=64)

        self.fusion_layer2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(64 + 64 + 64, 3, kernel_size=1),
            nn.Tanh()
        )

        self.cam = CloudAttentionMap(in_channels=3)

        if init_weights:
            self.apply(weights_init())

    @staticmethod
    def _make_gconv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        norm="in",
        activation="lrelu"
    ):
        """Create a gated convolution block."""
        return GatedConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=activation
        )

    @staticmethod
    def _upsample(x):
        """Upsample feature map by 2 times."""
        return F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear"
        )

    def _encode_optical(self, optical_image, block):
        """Encode optical image with SAR block guidance."""
        optical_features = {}

        optical_features["ec_0"] = torch.cat((optical_image, block), dim=1)
        optical_features["ec_1"] = self.optical_ec_1(optical_features["ec_0"])
        optical_features["ec_2"] = self.optical_ec_2(optical_features["ec_1"])
        optical_features["ec_3"] = self.optical_ec_3(optical_features["ec_2"])
        optical_features["ec_4"] = self.optical_ec_4(optical_features["ec_3"])

        return optical_features

    def _encode_sar(self, sar_edge):
        """Encode SAR edge features."""
        sar_features = {}

        sar_features["ec_0"] = sar_edge
        sar_features["ec_1"] = self.sar_ec_1(sar_features["ec_0"])
        sar_features["ec_2"] = self.sar_ec_2(sar_features["ec_1"])
        sar_features["ec_3"] = self.sar_ec_3(sar_features["ec_2"])
        sar_features["ec_4"] = self.sar_ec_4(sar_features["ec_3"])

        return sar_features

    def _decode_sar(self, sar_features, optical_features):
        """Decode SAR branch with optical skip features."""
        sar_decoded = sar_features["ec_4"]

        for layer_idx in range(4, 0, -1):
            sar_decoded = self._upsample(sar_decoded)

            skip_feature = optical_features[f"ec_{layer_idx - 1}"]
            sar_decoded = torch.cat((sar_decoded, skip_feature), dim=1)

            decoder = getattr(self, f"sar_dc_{layer_idx}")
            sar_decoded = decoder(sar_decoded)

        return sar_decoded

    def _decode_optical(self, optical_features, sar_features):
        """Decode optical branch with SAR skip features."""
        optical_decoded = optical_features["ec_4"]

        for layer_idx in range(4, 0, -1):
            optical_decoded = self._upsample(optical_decoded)

            skip_feature = sar_features[f"ec_{layer_idx - 1}"]
            optical_decoded = torch.cat((optical_decoded, skip_feature), dim=1)

            decoder = getattr(self, f"optical_dc_{layer_idx}")
            optical_decoded = decoder(optical_decoded)

        return optical_decoded

    def _fuse_features(self, optical_feature, sar_feature):
        """Fuse optical and SAR decoded features."""
        osgc_opt = self.osgc_opt(optical_feature, sar_feature)
        osgc_sar = self.osgc_sar(optical_feature, sar_feature)
        osgc_feature = torch.cat((osgc_opt, osgc_sar), dim=1)

        fused_feature = self.fusion_layer1(osgc_feature)

        attention_feature = self.ccfs_msfa(fused_feature, fused_feature)
        fused_feature = torch.cat((fused_feature, attention_feature), dim=1)
        fused_feature = self.fusion_layer2(fused_feature)

        fused_feature = self._upsample(fused_feature)
        output = self.out_layer(torch.cat((fused_feature, osgc_feature), dim=1))

        return output

    def forward(self, optical_image, block, sar_edge):
        optical_features = self._encode_optical(optical_image, block)
        sar_features = self._encode_sar(sar_edge)

        sar_decoded = self._decode_sar(sar_features, optical_features)
        optical_decoded = self._decode_optical(optical_features, sar_features)

        output = self._fuse_features(optical_decoded, sar_decoded)

        # Blend generated result and original image by cloud attention map.
        cloud_map = self.cam(output)
        output = output * cloud_map + optical_image * (1 - cloud_map)

        return output

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_ec_bn:
            for _, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
