import torch
import numpy as np
from lib.utils import ap_per_class, get_batch_statistics, non_max_suppression, xywh2xyxy


def tensor_to_numpy(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if isinstance(tensor, torch.autograd.Variable):
        tensor = tensor.data
    return tensor.numpy()


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

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


def _apply(fcn, acc, counter):
    result = {}
    for key, value in acc.items():
        if isinstance(value, dict):
            result[key] = _apply(fcn, value, counter.get(key, {}))
        else:
            result[key] = fcn(acc[key], counter.get(key, 0.0))
    return result


class _Accumulator():
    def __init__(self):
        self.numerator = {}
        self.denominator = {}

    def mean(self):
        return _apply(
            lambda n, d: n / d if d > 0 else 0.0,
            self.numerator, self.denominator)

    def update(self, data):
        self.numerator = _apply(
            lambda a, d: a[0] + d, data, self.numerator)
        self.denominator = _apply(
            lambda a, d: a[1] + d,
            data, self.denominator)


class SegMetrics:
    def __init__(self, num_classes, labels=None, ignore_index=-1, device=None):
        labels = labels or list(range(num_classes))
        self.labels = dict(enumerate(labels))
        self.accumulator = _Accumulator()
        self.num_classes = num_classes
        self.cm = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.long,
            requires_grad=False, device=device)
        self.ignore_index = ignore_index

    def metrics(self):
        metrics = {}
        metrics['accuracy'] = self._accuracy()
        metrics['class'] = {
            k: {'iou': v}
            for k, v in zip(self.labels, self._iou())
        }
        metrics['mIOU'] = np.mean(
            [c['iou'] for c in metrics['class'].values() if c['iou'] > 0.00])
        metrics.update(self.accumulator.mean())
        metrics = _apply(
            lambda x, y: float(x),
            metrics,
            {}
        )
        return metrics

    def _accuracy(self):
        accuracy = self.cm.diagonal().sum().float() / self.cm.sum().float()
        return tensor_to_numpy(accuracy)

    def _iou(self):
        colsum = self.cm.sum(dim=0)
        rowsum = self.cm.sum(dim=1)
        diag = self.cm.diagonal()
        iou = diag.float() / (colsum + rowsum - diag).float()
        return tensor_to_numpy(iou)

    def _confusion_matrix(self, outputs, targets):
        outputs = outputs.view(-1).contiguous()
        targets = targets.view(-1).contiguous()
        mask = (targets != self.ignore_index)
        comb = self.num_classes * outputs[mask] + targets[mask]
        comb = torch.bincount(comb, minlength=self.num_classes ** 2)
        return comb.reshape(self.num_classes, self.num_classes).long()

    def add(self, outputs, targets, loss):
        self.accumulator.update({'loss': (loss, 1)})
        self.cm += self._confusion_matrix(outputs, targets)


class DetectionMetrics:
    def __init__(self):
        self.labels = []
        self.sample_metrics = []  # List of tuples (TP, confs, pred)

    def compute_batch(self, yolo_outputs, targets, img_width, img_height,
                      conf_thres, nms_thres, iou_thres):
        if not targets[0] is None:
            # 获取标签中的类别
            self.labels += targets[:, 1].tolist()
            # 将标签中的xywh转换为xyxy
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, (2, 4)] *= img_width
            targets[:, (3, 5)] *= img_height

        # NMS
        yolo_outputs = non_max_suppression(
            yolo_outputs, conf_thres=conf_thres, iou_thres=nms_thres)
        if not targets[0] is None:
            # 若标签中有目标，则计算目标检测的KPI
            curr_metrics = get_batch_statistics(
                yolo_outputs, targets, iou_threshold=iou_thres)
            self.sample_metrics += curr_metrics
            return curr_metrics
        else:
            return None

    def metrics(self):
        if len(self.sample_metrics) == 0:
            return []
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*self.sample_metrics))]
        metrics_output = ap_per_class(
            true_positives, pred_scores, pred_labels, self.labels)

        precision, recall, AP, f1, ap_class = metrics_output
        evaluation_metrics = [
            ("validation/precision", precision.mean()),
            ("validation/recall", recall.mean()),
            ("validation/mAP", AP.mean()),
            ("validation/f1", f1.mean())]
        return evaluation_metrics
