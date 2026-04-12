import argparse
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, increment_path, non_max_suppression, scale_coords, set_logging, xywhn2xyxy
from utils.plots import plot_one_box
from utils.torch_utils import select_device


def infer_label_path(image_path, explicit_label_path=None):
    if explicit_label_path:
        return Path(explicit_label_path)

    parts = list(image_path.parts)
    if 'images' in parts:
        image_idx = parts.index('images')
        parts[image_idx] = 'labels'
        return Path(*parts).with_suffix('.txt')

    return image_path.with_suffix('.txt')


def load_ground_truth(label_path, image_shape):
    if not label_path.exists():
        return []

    boxes = []
    height, width = image_shape[:2]
    with label_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            values = line.split()
            if len(values) < 5:
                continue
            cls = int(float(values[0]))
            xywhn = np.array([[float(v) for v in values[1:5]]], dtype=np.float32)
            xyxy = xywhn2xyxy(xywhn, w=width, h=height)[0]
            boxes.append((cls, xyxy))
    return boxes


def main(opt):
    source = Path(opt.source)
    if not source.is_file():
        raise FileNotFoundError(f'image not found: {source}')

    label_path = infer_label_path(source, opt.label_path)

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / source.name

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    model = attempt_load(opt.weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(opt.img_size, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names

    if half:
        model.half()

    img0 = cv2.imread(str(source))
    if img0 is None:
        raise FileNotFoundError(f'failed to read image: {source}')

    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.half() if half else img_tensor.float()
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    with torch.no_grad():
        pred = model(img_tensor, augment=opt.augment)[0]

    pred = non_max_suppression(
        pred,
        opt.conf_thres,
        opt.iou_thres,
        classes=opt.classes,
        agnostic=opt.agnostic_nms,
    )[0]

    canvas = img0.copy()

    gt_boxes = load_ground_truth(label_path, img0.shape)
    for cls, xyxy in gt_boxes:
        cls_name = names[int(cls)] if names else str(cls)
        plot_one_box(
            xyxy,
            canvas,
            color=(0, 255, 0),
            label=f'GT {cls_name}' if opt.show_labels else None,
            line_thickness=2,
        )

    pred_count = 0
    if len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img0.shape).round()
        for *xyxy, conf, cls in pred.tolist():
            cls_name = names[int(cls)] if names else str(int(cls))
            label = f'PRED {cls_name} {conf:.2f}' if opt.show_labels else None
            plot_one_box(
                xyxy,
                canvas,
                color=(0, 0, 255),
                label=label,
                line_thickness=2,
            )
            pred_count += 1

    cv2.imwrite(str(save_path), canvas)

    print(f'image: {source}')
    print(f'weights: {opt.weights}')
    print(f'label file: {label_path if label_path.exists() else "not found"}')
    print(f'ground-truth boxes: {len(gt_boxes)}')
    print(f'predicted boxes: {pred_count}')
    print(f'saved overlay: {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on one image and save prediction + GT overlay.')
    parser.add_argument('--weights', required=True, type=str, help='checkpoint .pt path')
    parser.add_argument('--source', required=True, type=str, help='input image path')
    parser.add_argument('--label-path', type=str, default='', help='optional explicit YOLO label path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class id')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--show-labels', action='store_true', help='draw GT/pred text labels')
    parser.add_argument('--project', default='runs/overlay', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    main(parser.parse_args())
