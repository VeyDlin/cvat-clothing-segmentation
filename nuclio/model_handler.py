from PIL import Image
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
from u2net import U2NET
import requests
from urllib.parse import urlparse


class ModelHandler:
    checkpoint_url = "https://huggingface.co/VeyDlin/u2net_clothing_segmentation/resolve/main/u2net_clothing_segmentation.pth"

    def __init__(self, context):
        self.context = context
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context.logger.info(f"Device {self.device}")
        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        transforms_list += [NormalizeImage(0.5, 0.5)]
        self.transform_rgb = transforms.Compose(transforms_list)
        self.u2net = self.__load_u2net()

    def handle(self, image):
        self.context.logger.info(f"Handle {image.size[1]}, {image.size[0]}")
        
        # Resize the image to 512 pixels by the longest side
        resized_image, original_size = self.__resize_image(image, 512)
        
        # Transform the image to a tensor
        image_tensor = self.transform_rgb(resized_image)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        # Forward pass through the model
        output_tensor = self.u2net(image_tensor.to(self.device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
                                         
        # Upscale the mask to the original size
        output_tensor = self.__upscale_mask(output_tensor, original_size)

        return output_tensor

    def __resize_image(self, image, target_size):
        original_size = image.size  # (width, height)
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        return image, original_size

    def __upscale_mask(self, mask, original_size):
        # Note: mask is expected to have dimensions (1, height, width) after torch.max
        mask = F.interpolate(mask.float(), size=(original_size[1], original_size[0]), mode='bilinear', align_corners=False)
        mask = (mask > 0.5).float()  # Convert probabilities to binary mask
        return mask

    def __load_u2net(self):
        filename = urlparse(self.checkpoint_url).path.split('/')[-1]
        checkpoint_path = os.path.join("models", filename)

        if not os.path.exists(checkpoint_path):
            self.context.logger.info(f"First load checkpoint {checkpoint_path}")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            response = requests.get(self.checkpoint_url)
            if response.status_code == 200:
                with open(checkpoint_path, "wb") as file:
                    file.write(response.content)

        self.context.logger.info(f"Load model {filename}")
        model_state_dict = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.context.logger.info(f"Load model ok")

        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        net = U2NET(in_ch=3, out_ch=4)
        net.load_state_dict(new_state_dict)
        net = net.to(self.device)
        net = net.eval()
        self.context.logger.info(f"u2net ok")
        return net


class NormalizeImage(object):
    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normalization implemented only for 1, 3 and 18"
