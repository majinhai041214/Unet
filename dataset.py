import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class UNETDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform  #没有进行数据增强

        self.img_list = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        mask_path = os.path.join(self.mask_dir, self.img_list[index])

        image = Image.open(img_path).convert("L")  # 灰度图
        mask = Image.open(mask_path).convert("L")  # 掩码图也转灰度

        to_tensor = T.ToTensor()
        img = to_tensor(image)
        mask = to_tensor(mask)
        return img, mask
