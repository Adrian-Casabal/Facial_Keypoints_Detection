import torch.nn as nn
import timm


class KeypointCNN(nn.Module):
    def __init__(self, num_outputs=30, pretrained=True):
        super().__init__()

        try:
            self.backbone = timm.create_model(
                "hrnet_w18",
                pretrained=pretrained,
                in_chans=1,
                num_classes=0,
                global_pool="avg",
            )
            self.using_pretrained_weights = pretrained
        except Exception:
            self.backbone = timm.create_model(
                "hrnet_w18",
                pretrained=False,
                in_chans=1,
                num_classes=0,
                global_pool="avg",
            )
            self.using_pretrained_weights = False

        in_features = self.backbone.num_features
        self.regressor = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, num_outputs),
        )

        self._init_head()

    def _init_head(self):
        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.regressor.parameters()

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x
