import math
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
import logging
from misc.utils import judge_thresh


class RRdetector(nn.Module):
    def __init__(self, classifier, cls_norm, selfreweightCalibrate=True):
        super(RRdetector, self).__init__()
        self.classifier = classifier
        self.cls_norm = Normalize(*cls_norm)
        self.selfreweightCalibrate = selfreweightCalibrate
        self._thresh = None

    def __str__(self) -> str:
        stri = "[RRdetector] with "
        stri += "selfreweightCalibrate" if self.selfreweightCalibrate else ""
        return stri

    @property
    def thresh(self):
        return self._thresh

    @thresh.setter
    def thresh(self, value):
        logging.info("[RRdetector] Set thrs={} for detector".format(value))
        self._thresh = value

    def judge_distance(self, evi):
        return judge_thresh(evi, self._thresh)

    def inference_classifier(self, x):
        if x.min() < 0 or x.max() > 1:
            logging.warn("You may use a wrong data range.")
        norm_x = self.cls_norm(x)
        return self.classifier(norm_x)

    def get_score(self, out, out_aux):
        con_pre, _ = torch.softmax(out, dim=1).max(1)
        if self.selfreweightCalibrate:
            # bs x 1, Calibration function A \in [0,1]
            out_aux = out_aux.sigmoid().squeeze()
            evi = con_pre * out_aux
        else:
            raise NotImplementedError()
        return evi

    def _classify_helper(self, img_data, batch_size):
        pred_y = []
        with torch.no_grad():
            for idx in range(math.ceil(len(img_data) / batch_size)):
                start = idx * batch_size
                batch_data = img_data[start:start + batch_size].cuda()
                with torch.no_grad():
                    pred_y_batch, _ = self.inference_classifier(batch_data)
                    pred_y_batch = pred_y_batch.argmax(dim=1).cpu()
                pred_y.append(pred_y_batch)
        pred_y = torch.cat(pred_y, dim=0)
        return pred_y

    def classify_normal(self, img_data: torch.Tensor, batch_size: int):
        """Return prediction results of reformed data samples.

        Args:
            test_data ([type]): [description]
            batch_size ([type]): [description]

        Return:
            pred_y: prediction on original data
        """
        return self._classify_helper(img_data, batch_size)

    def detect(self, test_img: torch.Tensor, batch_size: int):
        all_pass, all_evi = [], torch.Tensor()
        with torch.no_grad():
            for idx in range(math.ceil(len(test_img) / batch_size)):
                start = idx * batch_size
                batch_data = test_img[start:start + batch_size].cuda()
                # perform detection
                out, out_aux = self.inference_classifier(batch_data)
                evi = self.get_score(out, out_aux).cpu()
                this_pass = self.judge_distance(evi)
                # collect results
                all_pass.append(this_pass)
                all_evi = torch.cat([all_evi, evi], dim=0)
        all_pass = torch.cat(all_pass, dim=0).long()
        return all_pass, all_evi

    def get_thrs(self, valid_loader, drop_rate=0.05):
        all_evi = []
        for img, _ in valid_loader:
            img = img.cuda()
            with torch.no_grad():
                out, out_aux = self.inference_classifier(img)
                evi = self.get_score(out, out_aux).cpu()
            all_evi.append(evi)
        all_evi = torch.cat(all_evi, dim=0)
        all_evi, _ = all_evi.sort()
        thrs = all_evi[int(len(all_evi) * drop_rate)].item()
        logging.info("[RRdetector] get thrs={}".format(thrs))
        return thrs

    def load_classifier(self, path, key="state_dict"):
        weight = torch.load(path)
        if key is not None:
            weight = weight[key]
        self.classifier.load_state_dict(weight)
        logging.info("[RRdetector] loaded classifier from: {}".format(path))
