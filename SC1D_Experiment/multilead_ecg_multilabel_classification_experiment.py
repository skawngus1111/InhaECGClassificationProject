import sys
from time import time

import torch

import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from utils.load_functions import load_model
from utils.calculate_metrics import specificity_score, get_scores
from ._SC1Dbase import BaseSignalClassificationExperiment

class SignalClassificationExperiment(BaseSignalClassificationExperiment):
    def __init__(self, args):
        super(SignalClassificationExperiment, self).__init__(args)

        self.threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])

        self.score_fun = {'Precision': precision_score, 'Recall': recall_score, 'Specificity': specificity_score, 'F1 score': f1_score}

    def fit(self):
        self.print_params()
        if self.args.train:
            for epoch in tqdm(range(self.args.start_epoch, self.args.final_epoch + 1)):
                print('\n============ EPOCH {}/{} ============\n'.format(epoch, self.args.final_epoch))
                if self.args.distributed: self.train_sampler.set_epoch(epoch)

                epoch_start_time = time()

                print("TRAINING")
                train_results = self.train_epoch(epoch)

                print("EVALUATE")
                val_results = self.val_epoch(epoch)

                self.history['train_loss'].append(train_results)
                self.history['val_loss'].append(val_results)

                total_epoch_time = time() - epoch_start_time
                m, s = divmod(total_epoch_time, 60)
                h, m = divmod(m, 60)

                print('\nEpoch {}/{} : train loss {} | val loss {} | current lr {} | took {} h {} m {} s'.format(
                    epoch, self.args.final_epoch, np.round(train_results, 4), np.round(val_results, 4),
                    self.current_lr(self.optimizer), int(h), int(m), int(s)))

            print("INFERENCE")
            test_results = self.inference(self.args.final_epoch)

            return self.model, self.optimizer, self.scheduler, self.history, test_results, self.metric_list
        else :
            print("INFERENCE")
            self.model = load_model(self.args, self.model)
            test_results = self.inference(self.args.final_epoch)

            return test_results, self.metric_list

    def train_epoch(self, epoch):
        self.model.train()

        total_loss, total = 0., 0

        for batch_idx, (signal, target) in enumerate(self.train_loader):
            loss, output, target = self.forward(signal, target, mode='train')
            self.backward(loss)
            total_loss += loss.item() * signal.size(0)
            total += signal.size(0)

            if (batch_idx + 1) % self.args.step == 0 or (batch_idx + 1) == len(self.train_loader):
                print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                    epoch, batch_idx + 1, len(self.train_loader), np.round((batch_idx + 1) / len(self.train_loader) * 100.0, 2),
                    total_loss / total
                ))

        train_loss = total_loss / total

        return train_loss

    def val_epoch(self, epoch):
        self.model.eval()

        total_loss, total = .0, 0

        with torch.no_grad():
            for batch_idx, (signal, target) in enumerate(self.test_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                loss, output, target = self.forward(signal, target, mode='val')

                total_loss += loss.item() * signal.size(0)
                total += signal.size(0)

        val_loss = total_loss / total

        return val_loss

    def inference(self, epoch):
        self.model.eval()

        total_loss, total = .0, 0
        y_true, y_pred = list(), list()

        with torch.no_grad():
            for batch_idx, (signal, target) in enumerate(self.test_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                self.start.record()
                loss, output, target = self.forward(signal, target, mode='val')

                self.end.record()
                torch.cuda.synchronize()
                self.inference_time_list.append(self.start.elapsed_time(self.end))

                for y_true_, y_pred_ in zip(target, output):
                    y_true.append(y_true_.cpu().detach().numpy())
                    y_pred.append((torch.sigmoid(y_pred_).cpu().detach().numpy() >= 0.5).astype(np.int_))

                total_loss += loss.item() * signal.size(0)
                total += signal.size(0)

        test_loss = total_loss / total
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        scores = get_scores(y_true, y_pred, self.score_fun)

        if self.args.final_epoch == epoch : print("Mean Inference Time (ms) : {} ({})".format(np.mean(self.inference_time_list), np.std(self.inference_time_list)))

        return test_loss, scores