import os
from PIL import Image
import torch
from torchvision import transforms
from glob import glob

class Dataset_maker(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(), # Scales data into [0,1] 
                transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
            ]
        )
        self.config = config
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(), # Scales data into [0,1] 
            ]
        )
        if is_train:
            if category:
                self.image_files = glob(
                    os.path.join(root, category, "train", "*", "*.jpg")
                )
            else:
                self.image_files = glob(
                    os.path.join(root, "train", "*", "*.jpg")
                )
        else:
            if category:
                self.image_files = glob(os.path.join(root, category, "test", "*", "*.jpg"))
            else:
                self.image_files = glob(os.path.join(root, "test", "*", "*.jpg"))
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        if(image.shape[0] == 1):
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
        
        # Xử lý khi là tập huấn luyện
        if self.is_train:
            label = 'good'
            return image, label
        else:
            if self.config.data.mask:
                # Cập nhật lại đường dẫn mask dựa trên dataset của bạn
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])  # Tạo mask trắng cho ảnh good
                    label = 'good'
                else:
                    # Đường dẫn cho ảnh defected
                    target_file = image_file.replace("/test/", "/ground_truth/").replace(".jpg", ".png")
                    target = Image.open(target_file)
                    target = self.mask_transform(target)
                    label = 'defective'
            else:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])  # Mask trắng cho ảnh good
                    label = 'good'
                else:
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])  # Mask cho ảnh defected
                    label = 'defective'

            return image, target, label

    def __len__(self):
        return len(self.image_files)
