import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def pixel_accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def pixel_accuracy_class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def mean_intersection_over_union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def intersection_over_union_crack(self):
        TP_1 = self.confusion_matrix[1, 1]
        FP_1 = self.confusion_matrix[0, 1]
        FN_1 = self.confusion_matrix[1, 0]
        # IoU = TP / (TP + FP + FN)
        IoU = TP_1 / (TP_1 + FP_1 + FN_1)
        return IoU

    def frequency_weighted_intersection_over_union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def precision(self):
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        precision_per_class = TP / (TP + FP)
        precision_per_class = np.nan_to_num(precision_per_class, nan=0.0)
        return precision_per_class

    def recall(self):
        TP = np.diag(self.confusion_matrix)
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        recall_per_class = TP / (TP + FN)
        recall_per_class = np.nan_to_num(recall_per_class, nan=0.0)
        return recall_per_class

    def f1_score(self):
        precisions = self.precision()
        recalls = self.recall()
        f1_per_class = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        return f1_per_class

    def mean_f1_score(self):
        # 计算平均F1分数 (mF1)
        return np.mean(self.f1_score())

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
