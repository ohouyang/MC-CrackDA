import numpy as np
from torch.utils.data import Dataset


class UDADataset(Dataset):
    DATA_STRATEGY = {
        "source": (lambda s, t: len(s), lambda idx, s, t: (idx, np.random.randint(0, len(t)))),
        "target": (lambda s, t: len(t), lambda idx, s, t: (np.random.randint(0, len(t)), idx))
    }

    def __init__(self, source, target, transform=None, data_strategy='target'):
        self.source = source
        self.target = target
        assert data_strategy in self.DATA_STRATEGY.keys(), 'Data_strategy not exist!'
        self.data_strategy = data_strategy
        self.transform = transform

    def _get_strategy_fun_list(self):
        return self.DATA_STRATEGY[self.data_strategy]

    def _get_strategy_id(self, idx):
        return self._get_strategy_fun_list()[1](idx, self.source, self.target)

    def _get_strategy_len(self):
        return self._get_strategy_fun_list()[0](self.source, self.target)

    def __len__(self):
        return self._get_strategy_len()

    def __getitem__(self, idx):
        id_source, idx_target = self._get_strategy_id(idx)
        source_img, source_mask = self.source[id_source]
        target_img, target_mask = self.target[idx_target]
        target_img_brightness = np.mean(target_img, axis=2)  # 图片明度

        if self.transform is not None:
            augmentation_source = self.transform(image=source_img, mask=source_mask)
            augmentation_target = self.transform(image=target_img, mask=target_mask,
                                                 img_brightness=target_img_brightness)
            source_img = augmentation_source['image']
            source_mask = augmentation_source['mask']
            target_img = augmentation_target['image']
            target_img_brightness = augmentation_target['img_brightness']
            target_mask = augmentation_target['mask']

        out = {'source_img': source_img, 'source_mask': source_mask,
               'target_img': target_img, 'target_mask': target_mask,
               'target_img_brightness': target_img_brightness}

        return out
