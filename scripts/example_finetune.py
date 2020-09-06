import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
import argparse

from rubiksnet.transforms import *
from rubiksnet.models import RubiksNet
from rubiksnet.shiftlib import RubiksShift2D, RubiksShiftBase


class ExampleTrainer:
    def __init__(
        self,
        num_classes,
        batch_size,
        gpu,
        lr,
        lr_shift_mult,
        momentum,
        weight_decay,
        total_epochs,
        pretrained_path,
    ):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = torch.device(gpu)
        self.pretrained_path = pretrained_path
        self.total_epochs = total_epochs
        self.train_dataloader = self.create_dataset_loader("train")
        self.test_dataloader = self.create_dataset_loader("test")
        self.model = self.create_model()
        self.optimizer = self.create_optimizer(
            lr=lr,
            lr_shift_mult=lr_shift_mult,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

    def create_model(self):
        net = RubiksNet.load_pretrained(self.pretrained_path)
        net.replace_new_fc(self.num_classes)
        net.to(self.device)
        return net

    def create_optimizer(self, lr, lr_shift_mult, momentum, weight_decay):
        shift_params = []
        regular_params = []
        for name, param in self.model.named_parameters():
            if name.endswith('shift'):
                shift_params.append(param)
            else:
                regular_params.append(param)

        param_groups = [
            {"params": shift_params, "lr": lr * lr_shift_mult},
            {"params": regular_params},
        ]
        return optim.SGD(
            param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    def create_dataset_loader(self, mode):
        dataset = ExampleVideoDatasets(
            num_classes=self.num_classes,
            transform=self.get_transforms(),
            total_size=32 * 50 if mode == "train" else 32 * 10,
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    def get_transforms(self):
        return [
            GroupMultiScaleCrop(256, [1, 0.875, 0.75, 0.66]),
            GroupRandomHorizontalFlip(),  # only for datasets that make sense
            GroupRandomCrop(224),
            Stack(),
            ToTorchFormatTensor(div=True),  # scale to [0, 1]
        ]

    def train_one_epoch(self, epoch):
        print(f"\nNew epoch: {epoch}")
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (video, targets) in enumerate(self.train_dataloader):
            video, targets = video.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(video)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(
                f"Epoch: {epoch+1}/{self.total_epochs} | "
                f"Batch: {batch_idx+1}/{len(self.train_dataloader)} | "
                f"Loss: {train_loss / (batch_idx + 1):.3f} | "
                f"Acc: {100.0 * correct / total:.2f}% ({correct}/{total})",
            )

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        print("Testing ...")
        with torch.no_grad():
            for batch_idx, (video, targets) in enumerate(self.test_dataloader):
                video, targets = video.to(self.device), targets.to(self.device)
                outputs = self.model(video)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                print(
                    f"Batch: {batch_idx+1}/{len(self.test_dataloader)} | "
                    f"Loss: {test_loss / (batch_idx + 1):.3f} | "
                    f"Acc: {100.0 * correct / total:.2f}% ({correct}/{total})",
                )

        print(
            f"\nFinal Acc for epoch {epoch}: "
            f"{100.0 * correct / total:.2f}% ({correct}/{total})"
        )

    def run(self):
        for epoch in range(self.total_epochs):
            self.train_one_epoch(epoch)
            self.test(epoch)


class ExampleVideoDatasets(data.Dataset):
    """
    This is only an example dataset.
    It generates a random video with an artificial label.
    You can replace this class with your custom dataset loader
    """

    def __init__(
        self, num_classes=50, num_frames=8, transform=None, total_size=32 * 50
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        if isinstance(transform, list):
            transform = torchvision.transforms.Compose(transform)
        self.transform = transform
        self.total_size = total_size

    def _load_dummy_frame(self, label):
        """
        Overwrite this function to load from another dataset
        """
        # randomly generate a frame
        dummy_value = label / self.num_classes
        img = dummy_value + np.random.randn(256, 256, 3) / self.num_classes / 10.0
        img *= 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def __getitem__(self, index):
        random_label = np.random.randint(0, self.num_classes)
        frames = [self._load_dummy_frame(random_label) for _ in range(self.num_frames)]
        if self.transform:
            frames = self.transform(frames)
        # frames should now be a 3D tensor: [T*C, H, W] for a single video
        return frames, random_label

    def __len__(self):
        """
        Calculate the total size of your dataset here
        """
        return self.total_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr-shift-mult",
        type=float,
        default=0.1,
        help="Shift layers typically need a lower learning rate. "
        "Good values are 0.1 or 0.01 * base LR",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument(
        "--pretrained-path", type=str, default="pretrained/kinetics_tiny.pth.tar"
    )
    args = parser.parse_args()

    trainer = ExampleTrainer(**vars(args))
    trainer.run()
