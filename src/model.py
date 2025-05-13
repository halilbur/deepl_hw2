import torch
import torch.nn as nn
import segmentation_models_pytorch as smp # <<< SMP kütüphanesini kullanıyoruz

# def get_smp_model(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation=None):
#     """
#     segmentation-models-pytorch kullanarak bir segmentasyon modeli oluşturur.
#     Args:
#         encoder_name (str): Kullanılacak encoder'ın adı (örn: 'resnet34', 'efficientnet-b0').
#         encoder_weights (str): Encoder için önceden eğitilmiş ağırlıklar ('imagenet' veya None).
#         in_channels (int): Giriş kanalı sayısı (KvasirMSBench için 3).
#         classes (int): Çıkış sınıfı sayısı (Binary segmentasyon için 1).
#         activation (str): Son katman aktivasyonu ('sigmoid' veya None).
#                            BCEWithLogitsLoss kullanıyorsak None olmalı.
#     Returns:
#         torch.nn.Module: Oluşturulan SMP modeli.
#     """
#     print(f"SMP Modeli Oluşturuluyor: Encoder={encoder_name}, Ağırlıklar={encoder_weights}, Giriş={in_channels}, Çıkış={classes}, Aktivasyon={activation}")
#     # U-Net veya başka bir mimari seçilebilir (FPN, DeepLabV3+, vb.)
#     model = smp.Unet(
#         encoder_name=encoder_name,
#         encoder_weights=encoder_weights,
#         in_channels=in_channels,
#         classes=classes,
#         activation=activation,
#     )
#     return model
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else: # Ensure shortcut still passes through if in_channels == out_channels
             self.shortcut = nn.Identity()


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut_out = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += shortcut_out
        out = self.relu2(out)
        return out

class SimpleCNNEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1_res_block = ResidualConvBlock(in_channels, 64)    # 256x256, 64ch
        self.enc2_res_block = ResidualConvBlock(64, 128)             # 128x128, 128ch
        self.enc3_res_block = ResidualConvBlock(128, 256)            # 64x64,   256ch
        self.enc4_res_block = ResidualConvBlock(256, 512)            # 32x32,   512ch
        self.bottleneck_res_block = ResidualConvBlock(512, 1024)     # 16x16,   1024ch

    def forward(self, x):
        skip_connections = []

        s1_out = self.enc1_res_block(x)
        skip_connections.append(s1_out) # Skip from 256x256, 64ch
        x_pooled1 = self.pool(s1_out)

        s2_out = self.enc2_res_block(x_pooled1)
        skip_connections.append(s2_out) # Skip from 128x128, 128ch
        x_pooled2 = self.pool(s2_out)

        s3_out = self.enc3_res_block(x_pooled2)
        skip_connections.append(s3_out) # Skip from 64x64, 256ch
        x_pooled3 = self.pool(s3_out)

        s4_out = self.enc4_res_block(x_pooled3)
        skip_connections.append(s4_out) # Skip from 32x32, 512ch
        x_pooled4 = self.pool(s4_out)
        
        bottleneck = self.bottleneck_res_block(x_pooled4) # 16x16, 1024ch

        return bottleneck, skip_connections[::-1] # Skips: [s4, s3, s2, s1]

class UNetDecoder(nn.Module):
    def __init__(self, num_classes=1): # num_classes is 1 for binary segmentation
        super().__init__()
        
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1_res_block = ResidualConvBlock(1024, 512) # Takes 512 (upsampled) + 512 (skip s4)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2_res_block = ResidualConvBlock(512, 256) # Takes 256 (upsampled) + 256 (skip s3)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_res_block = ResidualConvBlock(256, 128) # Takes 128 (upsampled) + 128 (skip s2)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4_res_block = ResidualConvBlock(128, 64) # Takes 64 (upsampled) + 64 (skip s1)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, bottleneck, skip_connections):
        # skip_connections are [s4_out, s3_out, s2_out, s1_out]
        x = self.upconv1(bottleneck) 
        x = torch.cat([x, skip_connections[0]], dim=1) 
        x = self.dec1_res_block(x) 

        x = self.upconv2(x) 
        x = torch.cat([x, skip_connections[1]], dim=1) 
        x = self.dec2_res_block(x) 

        x = self.upconv3(x) 
        x = torch.cat([x, skip_connections[2]], dim=1) 
        x = self.dec3_res_block(x)

        x = self.upconv4(x) 
        x = torch.cat([x, skip_connections[3]], dim=1) 
        x = self.dec4_res_block(x) 

        output_mask = self.out_conv(x) 
        return output_mask

class CustomUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.encoder = SimpleCNNEncoder(in_channels=in_channels)
        self.decoder = UNetDecoder(num_classes=num_classes)

    def forward(self, x):
        bottleneck, skip_connections = self.encoder(x)
        segmentation_mask = self.decoder(bottleneck, skip_connections)
        return segmentation_mask
