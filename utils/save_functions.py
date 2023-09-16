import os

import torch

import numpy as np
import pandas as pd

from .get_functions import get_save_path
from .plot_functions import plot_loss_acc

def save_result(args, model, optimizer, scheduler, history, test_results, current_epoch):
    if (args.distributed and torch.distributed.get_rank() == 0) or not args.multiprocessing_distributed:
        model_dirs = get_save_path(args)

        print("Your experiment is saved in {}.".format(model_dirs))

        print("STEP1. Save {} Model Weight...".format(args.model_name))
        save_model(model, optimizer, scheduler, model_dirs, current_epoch)

        print("STEP2. Save {} Model Test Results...".format(args.model_name))
        # if type(test_results) is list:
        save_metrics(test_results, model_dirs, current_epoch)

        if args.final_epoch == current_epoch:
            print("STEP3. Save {} Model History...".format(args.model_name))
            save_loss(history, model_dirs)

            # print("STEP4. Plot {} Model History...".format(args.model_name))
            # plot_loss_acc(history, model_dirs)

        print("Current EPOCH {} model is successfully saved at {}".format(current_epoch, model_dirs))

def save_model(model, optimizer, scheduler, model_dirs, current_epoch):
    check_point = {
        'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'current_epoch': current_epoch
    }

    torch.save(check_point, os.path.join(model_dirs, 'model_weights/model_weight(EPOCH {}).pth.tar'.format(current_epoch)))

def save_metrics(test_results, model_dirs, current_epoch):
    print("###################### TEST REPORT ######################")
    for metric in test_results.keys():
        print("+++++++++++++++++++++++++ Metric = {} +++++++++++++++++++++++++".format(metric))
        for label in test_results[metric].keys():
            print("Label {} Mean {}    :\t {}".format(label, metric, np.round(test_results[metric][label], 4)))
    print("###################### TEST REPORT ######################\n")

    test_results_save_path = os.path.join(model_dirs, 'test_reports', 'test_report(EPOCH {}).txt'.format(current_epoch))

    f = open(test_results_save_path, 'w')

    f.write("###################### TEST REPORT ######################\n")
    for metric in test_results.keys():
        f.write("+++++++++++++++++++++++++ Metric = {} +++++++++++++++++++++++++\n".format(metric))
        for label in test_results[metric].keys():
            f.write("Label {} Mean {}    :\t {}\n".format(label, metric, np.round(test_results[metric][label], 4)))
        f.write("\n")
    f.write("###################### TEST REPORT ######################\n\n")

    f.close()

    print("test results txt file is saved at {}".format(test_results_save_path))

def save_loss(history, model_dirs):
    pd.DataFrame(history).to_csv(os.path.join(model_dirs, 'loss.csv'), index=False)