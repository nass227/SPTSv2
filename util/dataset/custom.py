from pathlib import Path
import math
import random
import torch
import torchvision
import numpy as np
from torch.utils.data import ConcatDataset
import datasets.sptsv2_transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, dataset_name, max_length):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, dataset_name, max_length)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img1, target1 = self._transforms(img, target)
            img2, target2 = self._transforms(img, target)
        else:
            img1, target1 = img, target
            img2, target2 = img, target
        return img1, img2, target1, target2


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, dataset_name='', max_length=25):
        self.return_masks = return_masks
        self.dataset_name = dataset_name
        self.max_length = max_length

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64) if len(classes) > 0 else torch.zeros((0,), dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0]) if boxes.shape[0] > 0 else torch.zeros((0,), dtype=torch.bool)
        boxes = boxes[keep]
        classes = classes[keep]

        target_out = {}
        target_out["boxes"] = boxes
        target_out["labels"] = classes
        target_out["image_id"] = image_id
        target_out["dataset_name"] = self.dataset_name

        area = torch.tensor([obj["area"] for obj in anno], dtype=torch.float32) if len(anno) > 0 else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno], dtype=torch.int64) if len(anno) > 0 else torch.zeros((0,), dtype=torch.int64)
        target_out["area"] = area[keep]
        target_out["iscrowd"] = iscrowd[keep]

        target_out["orig_size"] = torch.as_tensor([int(h), int(w)])
        target_out["size"] = torch.as_tensor([int(h), int(w)])

        recog = [obj['rec'][:self.max_length] for obj in anno]
        if len(recog) > 0:
            recog = torch.tensor(recog, dtype=torch.long).reshape(-1, self.max_length)
        else:
            recog = torch.zeros((0, self.max_length), dtype=torch.long)
        target_out["rec"] = recog[keep]

        bezier_pts = [obj['bezier_pts'] for obj in anno]
        if len(bezier_pts) > 0:
            bezier_pts = torch.tensor(bezier_pts, dtype=torch.float32).reshape(-1, 16)
        else:
            bezier_pts = torch.zeros((0, 16), dtype=torch.float32)
        target_out['bezier_pts'] = bezier_pts[keep]

        center_pts = torch.zeros(bezier_pts.shape[0], 2, dtype=torch.float32)
        for i in range(bezier_pts.shape[0]):
            tmp = bezier_pts[i]
            if tmp[-1].item() == 0 and tmp[-2].item() == 0:
                xc = 0
                yc = 0
                count = 0
                tmp2 = tmp.view(-1, 2)
                for j in range(tmp2.shape[0]):
                    if tmp2[j][0].item() == 0 and tmp2[j][1].item() == 0:
                        continue
                    xc += tmp2[j][0].item()
                    yc += tmp2[j][1].item()
                    count += 1
                if count > 0:
                    xc = xc / count
                    yc = yc / count
            else:
                polygon = bezier_to_polygon(tmp)
                length = int(len(polygon) / 2)
                top = torch.tensor(polygon[:length], dtype=torch.float32)
                bottom = torch.tensor(polygon[length:], dtype=torch.float32)

                count = 0
                x1 = 0
                y1 = 0
                for j in range(length):
                    if top[j][0] == 0 and top[j][1] == 0:
                        continue
                    x1 += top[j][0].item()
                    y1 += top[j][1].item()
                    count += 1
                if count > 0:
                    xt = x1 / count
                    yt = y1 / count
                else:
                    xt = 0
                    yt = 0

                count = 0
                x2 = 0
                y2 = 0
                for j in range(length):
                    if bottom[j][0] == 0 and bottom[j][1] == 0:
                        continue
                    x2 += bottom[j][0].item()
                    y2 += bottom[j][1].item()
                    count += 1
                if count > 0:
                    xb = x2 / count
                    yb = y2 / count
                else:
                    xb = 0
                    yb = 0

                xc = (xt + xb) / 2
                yc = (yt + yb) / 2

            center_pts[i][0] = xc
            center_pts[i][1] = yc

        target_out['center_pts'] = center_pts[keep]
        assert target_out['center_pts'].shape[0] == target_out['bezier_pts'].shape[0]

        return image, target_out


def dynamic_point(pts, radius):
    theta = random.uniform(0, 1) * 2 * math.pi
    x_new = pts[0] + radius * math.cos(theta)
    y_new = pts[1] - radius * math.sin(theta)
    return x_new, y_new


def bezier_to_polygon(bezier):
    u = np.linspace(0, 1, 20)
    bezier = np.array(bezier)
    bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
        + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
        + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
        + np.outer(u ** 3, bezier[:, 3])

    points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
    return points.tolist()


def make_coco_transforms(image_set, max_size_train, min_size_train, max_size_test, min_size_test,
                         crop_min_ratio, crop_max_ratio, crop_prob, rotate_max_angle, rotate_prob,
                         brightness, contrast, saturation, hue, distortion_prob):

    transforms = []
    if image_set == 'train':
        transforms.append(T.RandomSizeCrop(crop_min_ratio, crop_max_ratio, True, crop_prob))
        transforms.append(T.RandomRotate(rotate_max_angle, rotate_prob))
        transforms.append(T.RandomResize(min_size_train, max_size_train))
        transforms.append(T.RandomDistortion(brightness, contrast, saturation, hue, distortion_prob))
    elif image_set == 'val':
        transforms.append(T.RandomResize([min_size_test], max_size_test))

    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(None, None))

    return T.Compose(transforms)


def build(image_set, args):
    root = Path(args.data_root)

    if image_set == 'train':
        dataset_names = args.train_dataset.split(':')
    elif image_set == 'val':
        dataset_names = args.val_dataset.split(':')
    else:
        raise ValueError(f"Unsupported image_set: {image_set}")

    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'custom_train':
            img_folder = root / "custom" / "train_images"
            ann_file = root / "custom" / "train.json"

        elif dataset_name == 'custom_val':
            img_folder = root / "custom" / "test_images"
            ann_file = root / "custom" / "test.json"

        if dataset_name == 'ICDAR2019_train':
            img_folder = root / "ICDAR2019" / "train_imgs"
            ann_file = root / "ICDAR2019" / "train.json"

        elif dataset_name == 'ICDAR2019_test':
            img_folder = root / "ICDAR2019" / "test_imgs"
            ann_file = root / "ICDAR2019" / "test.json"

        else:
            raise NotImplementedError(f"Unknown dataset_name: {dataset_name}")

        transforms = make_coco_transforms(
            image_set,
            args.max_size_train,
            args.min_size_train,
            args.max_size_test,
            args.min_size_test,
            args.crop_min_ratio,
            args.crop_max_ratio,
            args.crop_prob,
            args.rotate_max_angle,
            args.rotate_prob,
            args.brightness,
            args.contrast,
            args.saturation,
            args.hue,
            args.distortion_prob
        )

        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=transforms,
            return_masks=args.masks,
            dataset_name=dataset_name,
            max_length=args.max_length
        )
        datasets.append(dataset)

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)