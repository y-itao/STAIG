import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from byol_pytorch import BYOL
from torchvision import models
from PIL import Image
import os
import numpy as np
import torchvision.transforms as T
import random
from torch import nn
import time


def process_images(dataset_name, slide, epoch_num):
    path = f"./Dataset/{dataset_name}/{slide}/clip_image_filter"

    torch.manual_seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(12345)
    random.seed(12345)

    class CustomDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_list = sorted(os.listdir(root_dir), key=lambda x: int(x.split('.')[0]))

        def __len__(self):
            return len(self.image_list)

        def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, self.image_list[idx])
            img = Image.open(img_name).convert('RGB')

            if self.transform:
                img = self.transform(img)

            return img

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=2)

    class RandomApply(nn.Module):
        def __init__(self, fn, p):
            super().__init__()
            self.fn = fn
            self.p = p

        def forward(self, x):
            if random.random() > self.p:
                return x
            return self.fn(x)

    DEFAULT_AUG = torch.nn.Sequential(
        RandomApply(
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            p=0.3
        ),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomApply(
            T.GaussianBlur((3, 3), (1.0, 2.0)),
            p=0.2
        ),
        T.RandomRotation(degrees=(0, 360)),
        T.RandomResizedCrop((256, 256)),
        T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])),

    )

    learner = BYOL(
            models.resnet50(pretrained=True),
            image_size=256,
            hidden_layer='avgpool',
            augment_fn=DEFAULT_AUG
        )
    if torch.cuda.is_available():
        learner = learner.cuda()

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    print('start training!')
    for epoch in range(epoch_num):
        start_time = time.time()
        for images in data_loader:
            images = images.cuda() if torch.cuda.is_available() else images
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f} seconds')

    torch.save(learner.state_dict(), 'learner.pth')

    learner.eval()
    embeddings = []
    print('start eval!')
    for i in range(len(dataset)):
        img = dataset[i]
        img = img.cuda() if torch.cuda.is_available() else img
        with torch.no_grad():
            _, embedding = learner(img.unsqueeze(0), return_embedding=True)
            embeddings.append(embedding.cpu().numpy())

    embeddings = np.vstack(embeddings)
    np.save(f'./Dataset/{dataset_name}/{slide}/embeddings.npy', embeddings)
    print(embeddings.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and extract embeddings with BYOL.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--slide", type=str, required=True, help="Slide name")
    parser.add_argument("--epoch_num", type=int, required=True, help="Number of epochs")

    args = parser.parse_args()

    process_images(args.dataset, args.slide, args.epoch_num)