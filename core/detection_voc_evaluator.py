import copy
import numpy as np
from chainer import reporter
import chainer.training.extensions
from core.eval_detection_voc import eval_detection_voc
from core.apply_to_iterator import apply_to_iterator

class DetectionVOCEvaluator(chainer.training.extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, iterator, target, use_07_metric=False,
            label_names=None, comm=None):
        if iterator is None:
            iterator = {}
        super(DetectionVOCEvaluator, self).__init__(
            iterator, target)
        self.use_07_metric = use_07_metric
        self.label_names = label_names
        self.comm = comm

    def evaluate(self):
        target = self._targets['main']
        if self.comm is not None and self.comm.rank != 0:
            apply_to_iterator(target.predict, None, comm=self.comm)
            return {}
        iterator = self._iterators['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        in_values, out_values, rest_values = apply_to_iterator(
            target.predict, it, comm=self.comm)
        # delete unused iterators explicitly
        del in_values

        pred_bboxes, pred_labels, pred_scores = out_values

        if len(rest_values) == 3:
            gt_bboxes, gt_labels, gt_difficults = rest_values
        elif len(rest_values) == 2:
            gt_bboxes, gt_labels = rest_values
            gt_difficults = None

        result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=self.use_07_metric)

        report = {'map': result['map']}

        if self.label_names is not None:
            for l, label_name in enumerate(self.label_names):
                try:
                    report['ap/{:s}'.format(label_name)] = result['ap'][l]
                except IndexError:
                    report['ap/{:s}'.format(label_name)] = np.nan

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation
