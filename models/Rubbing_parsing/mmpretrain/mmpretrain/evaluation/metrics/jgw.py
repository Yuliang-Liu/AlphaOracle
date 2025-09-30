# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


def _precision_recall_f1_support(pred_positive, gt_positive, average):
    """calculate base classification task metrics, such as  precision, recall,
    f1_score, support."""
    average_options = ['micro', 'macro', None]
    assert average in average_options, 'Invalid `average` argument, ' \
                                       f'please specify from {average_options}.'

    # ignore -1 target such as difficult sample that is not wanted
    # in evaluation results.
    # only for calculate multi-label without affecting single-label behavior
    ignored_index = gt_positive == -1
    pred_positive[ignored_index] = 0
    gt_positive[ignored_index] = 0

    class_correct = (pred_positive & gt_positive)
    if average == 'micro':
        tp_sum = class_correct.sum()
        pred_sum = pred_positive.sum()
        gt_sum = gt_positive.sum()
    else:
        tp_sum = class_correct.sum(0)
        pred_sum = pred_positive.sum(0)
        gt_sum = gt_positive.sum(0)

    precision = tp_sum / torch.clamp(pred_sum, min=1).float() * 100
    recall = tp_sum / torch.clamp(gt_sum, min=1).float() * 100
    f1_score = 2 * precision * recall / torch.clamp(
        precision + recall, min=torch.finfo(torch.float32).eps)
    if average in ['macro', 'micro']:
        precision = precision.mean(0)
        recall = recall.mean(0)
        f1_score = f1_score.mean(0)
        support = gt_sum.sum(0)
    else:
        support = gt_sum
    return precision, recall, f1_score, support


@METRICS.register_module()
class JGWMetric(BaseMetric):
    r"""Accuracy evaluation metric.

    For either binary classification or multi-class classification, the
    accuracy is the fraction of correct predictions in all predictions:

    .. math::

        \text{Accuracy} = \frac{N_{\text{correct}}}{N_{\text{all}}}

    Args:
        topk (int | Sequence[int]): If the ground truth label matches one of
            the best **k** predictions, the sample will be regard as a positive
            prediction. If the parameter is a tuple, all of top-k accuracy will
            be calculated and outputted together. Defaults to 1.
        thrs (Sequence[float | None] | float | None): If a float, predictions
            with score lower than the threshold will be regard as the negative
            prediction. If None, not apply threshold. If the parameter is a
            tuple, accuracy based on all thresholds will be calculated and
            outputted together. Defaults to 0.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:
        >>> import torch
        >>> from mmpretrain.evaluation import Accuracy
        >>> # -------------------- The Basic Usage --------------------
        >>> y_pred = [0, 2, 1, 3]
        >>> y_true = [0, 1, 2, 3]
        >>> Accuracy.calculate(y_pred, y_true)
        tensor([50.])
        >>> # Calculate the top1 and top5 accuracy.
        >>> y_score = torch.rand((1000, 10))
        >>> y_true = torch.zeros((1000, ))
        >>> Accuracy.calculate(y_score, y_true, topk=(1, 5))
        [[tensor([9.9000])], [tensor([51.5000])]]
        >>>
        >>> # ------------------- Use with Evalutor -------------------
        >>> from mmpretrain.structures import DataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_samples = [
        ...     DataSample().set_gt_label(0).set_pred_score(torch.rand(10))
        ...     for i in range(1000)
        ... ]
        >>> evaluator = Evaluator(metrics=Accuracy(topk=(1, 5)))
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(1000)
        {
            'accuracy/top1': 9.300000190734863,
            'accuracy/top5': 51.20000076293945
        }
    """
    default_prefix: Optional[str] = 'accuracy'

    def __init__(self,
                 dl,
                 topk: Union[int, Sequence[int]] = (1,),
                 thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.dl = dl

        if isinstance(topk, int):
            self.topk = (topk,)
        else:
            self.topk = tuple(topk)

        if isinstance(thrs, float) or thrs is None:
            self.thrs = (thrs,)
        else:
            self.thrs = tuple(thrs)

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            if 'pred_score' in data_sample:
                result['pred_score'] = data_sample['pred_score'].cpu()
            else:
                result['pred_label'] = data_sample['pred_label'].cpu()
            result['gt_label'] = data_sample['gt_label'].cpu()
            # Save the result to `self.results`.
            result['name'] = data_sample['img_path'].split('/')[-1]
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = {}
        names = []
        for res in results:
            names.append(res['name'])
        # concat
        target = torch.cat([res['gt_label'] for res in results])
        if 'pred_score' in results[0]:
            pred = torch.stack([res['pred_score'] for res in results])

            try:
                acc = self.calculate(self, pred, target, self.topk, self.thrs, names)
            except ValueError as e:
                # If the topk is invalid.
                raise ValueError(
                    str(e) + ' Please check the `val_evaluator` and '
                             '`test_evaluator` fields in your config file.')

            multi_thrs = len(self.thrs) > 1
            for i, k in enumerate(self.topk):
                for j, thr in enumerate(self.thrs):
                    name = f'top{k}'
                    if multi_thrs:
                        name += '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                    metrics[name] = acc[i][j].item()
        else:
            # If only label in the `pred_label`.
            pred = torch.cat([res['pred_label'] for res in results])
            acc = self.calculate(self, pred, target, self.topk, self.thrs, names)
            metrics['top1'] = acc.item()

        return metrics

    @staticmethod
    def calculate(self,
                  pred: Union[torch.Tensor, np.ndarray, Sequence],
                  target: Union[torch.Tensor, np.ndarray, Sequence],
                  topk: Sequence[int] = (1,),
                  thrs: Sequence[Union[float, None]] = (0.,),
                  names=None,
                  ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """Calculate the accuracy.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            thrs (Sequence[float | None]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. None means no thresholds.
                Defaults to (0., ).
            thrs (Sequence[float]): Predictions with scores under
                the thresholds are considered negative. It's only used
                when ``pred`` is scores. Defaults to (0., ).
            names

        Returns:
            torch.Tensor | List[List[torch.Tensor]]: Accuracy.

            - torch.Tensor: If the ``pred`` is a sequence of label instead of
              score (number of dimensions is 1). Only return a top-1 accuracy
              tensor, and ignore the argument ``topk` and ``thrs``.
            - List[List[torch.Tensor]]: If the ``pred`` is a sequence of score
              (number of dimensions is 2). Return the accuracy on each ``topk``
              and ``thrs``. And the first dim is ``topk``, the second dim is
              ``thrs``.
        """

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        num = pred.size(0)
        lable = self.dataset_meta['classes']
        gt_lable = []
        for i in range(len(target)):
            gt_lable.append(lable[int(target[i])])

        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match " \
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            # For pred label, ignore topk and acc
            pred_label = pred.int()
            correct = pred.eq(target).float().sum(0, keepdim=True)
            acc = correct.mul_(100. / num)
            return acc
        else:
            # For pred score, calculate on all topk and thresholds.
            pred = pred.float()
            maxk = max(topk)

            if maxk > pred.size(1):
                raise ValueError(
                    f'Top-{maxk} accuracy is unavailable since the number of '
                    f'categories is {pred.size(1)}.')

            pred_score, pred_label = pred.topk(maxk, dim=1)
            pred_label = pred_label.t()
            correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
            results = []
            for k in topk:
                results.append([])
                for thr in thrs:
                    # Only prediction values larger than thr are counted
                    # as correct
                    _correct = correct
                    if thr is not None:
                        _correct = _correct & (pred_score.t() > thr)
                    correct_k = _correct[:k].reshape(-1).float().sum(
                        0, keepdim=True)
                    acc = correct_k.mul_(100. / num)
                    results[-1].append(acc)

            with open(self.dl, 'w') as f:
                f.write('The accuracy of top 1 is: {}\n'.format(float(results[0][0])))
                f.write('The accuracy of top {} is: {}\n'.format(maxk, float(results[1][0])))
                for i in range(len(target)):
                    f.write('\n')
                    a = pred[i].numpy()
                    b = a.argsort()[-max(topk):][::-1]
                    f.write('Class: {}:\n'.format(gt_lable[i]))
                    f.write('Name: {}:\n'.format(names[i]))
                    for j in range(max(topk)):
                        f.write('The probability of prediction {} is: {}\n'.format(lable[b[j]], a[b[j]]))

            return results


