import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=21,                     # model output channels (number of classes in your dataset)
)

# print(smp.encoders.get_encoder_names())
print(smp.encoders.get_preprocessing_fn(encoder_name="resnet50", pretrained="imagenet"))