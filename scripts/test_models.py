import torch
import argparse
import time
from sklearn.metrics import confusion_matrix

from rubiksnet.dataset import RubiksDataset, return_dataset
from rubiksnet.models import RubiksNet
from rubiksnet.transforms import *


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main():
    assert torch.cuda.is_available(), "CUDA must be available to run this example"
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        description="RubiksNet testing on the full validation set"
    )
    parser.add_argument("dataset", type=str)
    parser.add_argument(
        "-p", "--pretrained", type=str, required=True, help="pretrained checkpoint path"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default="./",
        help="we assume the dataset to be located at <root_path>/<dataset_name>",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="number of video frames to be passed to the network as a single clip",
    )
    parser.add_argument(
        "--two-clips",
        action="store_true",
        help='enable "two clip evaluation" protocol.',
    )
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 8)",
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    args = parser.parse_args()

    (num_classes, args.train_list, val_list, root_path, prefix,) = return_dataset(
        args.dataset, args.root_path
    )
    print(f"=> dataset: {args.dataset}")
    print(f"=> root_path: {args.root_path}")
    print(f"=> num_classes: {num_classes}")

    net = RubiksNet.load_pretrained(args.pretrained)
    print(f"=> tier: {net.tier}")
    print(f"=> variant: {net.variant}")
    param = sum(x.numel() for x in net.parameters())
    print(f"=> param: {param / 1e6:.1f}M")

    if args.two_clips:
        twice_sample = True
        test_crops = 3
    else:
        twice_sample = False
        test_crops = 1
    print(f"=> eval mode: {'2-clip' if args.two_clips else '1-clip'}")

    if test_crops == 1:
        cropping = torchvision.transforms.Compose(
            [GroupScale(net.scale_size), GroupCenterCrop(net.input_size),]
        )
    elif test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose(
            [GroupFullResSample(net.input_size, net.scale_size, flip=False)]
        )
    elif test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose(
            [GroupOverSample(net.input_size, net.scale_size, flip=False)]
        )
    elif test_crops == 10:
        cropping = torchvision.transforms.Compose(
            [GroupOverSample(net.input_size, net.scale_size)]
        )
    else:
        raise ValueError(
            f"Only 1, 5, 10 crops are supported while we got {test_crops}."
        )

    data_loader = torch.utils.data.DataLoader(
        RubiksDataset(
            root_path,
            val_list,
            num_segments=args.frames,
            new_length=1,
            image_tmpl=prefix,
            test_mode=True,
            remove_missing=True,
            transform=torchvision.transforms.Compose(
                [
                    cropping,
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize(net.input_mean, net.input_std),
                ]
            ),
            dense_sample=False,
            twice_sample=twice_sample,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.gpus is None:
        args.gpus = list(range(torch.cuda.device_count()))

    net = torch.nn.DataParallel(net.cuda(args.gpus[0]), device_ids=args.gpus)
    net.eval()

    output = []
    proc_start_time = time.time()

    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            batch_size = label.numel()
            num_crop = test_crops

            if twice_sample:
                num_crop *= 2

            data_in = data.view(-1, 3, data.size(2), data.size(3))
            data_in = data_in.view(
                batch_size * num_crop, args.frames, 3, data_in.size(2), data_in.size(3),
            )
            rst = net(data_in)
            rst = rst.reshape(batch_size, num_crop, -1).mean(1)
            rst = rst.data.cpu().numpy().copy()
            rst = rst.reshape(batch_size, num_classes)

            for p, g in zip(rst, label.cpu().numpy()):
                output.append([p[None, ...], g])
            cnt_time = time.time() - proc_start_time
            prec1, prec5 = accuracy(torch.from_numpy(rst), label, topk=(1, 5))
            top1.update(prec1.item(), label.numel())
            top5.update(prec5.item(), label.numel())
            if i % 20 == 0:
                print(
                    f"video {i * args.batch_size} done, total {i * args.batch_size}/{len(data_loader.dataset)}, "
                    f"average {float(cnt_time) / (i+1) / args.batch_size:.3f} sec/video, "
                    f"moving Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}"
                )

    video_pred = [np.argmax(x[0]) for x in output]
    video_pred_top5 = [
        np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:5] for x in output
    ]
    video_labels = [x[1] for x in output]

    cf = confusion_matrix(video_labels, video_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt

    print("\n====================== Evaluation Complete ======================")
    print("Class confusion matrix:")
    print(cls_acc)

    print(f"\nAccuracy: top 1: {top1.avg:.02f}%\ttop 5: {top5.avg:.02f}%")


if __name__ == "__main__":
    main()
