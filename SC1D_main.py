import os
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse

from SC1D_Experiment.multilead_ecg_multilabel_classification_experiment import SignalClassificationExperiment
# from SC1D_Experiment.multilead_ecgplot_multilabel_classification_experiment import SignalPlotClassificationExperiment
from utils.save_functions import save_metrics, save_result
from utils.get_functions import get_save_path

def SC1D_main(args) :
    print("Hello! We start experiment for 1D Signal Classification!")
    print("Distributed Data Parallel {}".format(args.multiprocessing_distributed))

    try:
        dataset_dir = os.path.join(args.data_path, args.data_type)
    except TypeError:
        print("join() argument must be str, bytes, or os.PathLike object, not 'NoneType'")
        print("Please explicitely write the dataset type")
        sys.exit()

    args.dataset_dir = dataset_dir
    if args.data_type == '12ECG':
        args.seq_length = 4096
        args.num_channels = 12
        args.num_classes = 6
    elif args.data_type == '12ECGPaper':
        args.image_size = 512
        args.num_channels = 3
        args.num_classes = 6

    args.distributed = False
    if args.multiprocessing_distributed and args.train:
        args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    else :
        experiment = SignalClassificationExperiment(args)
        if args.train:
            model, optimizer, scheduler, history, test_results, metric_list = experiment.fit()
            save_result(args, model, optimizer, scheduler, history, test_results, args.final_epoch, metric_list)
        else:
            test_results, metric_list = experiment.fit()
            model_dirs = get_save_path(args)

            print("Save {} Model Test Results...".format(args.model_name))
            save_metrics(test_results, model_dirs, args.final_epoch, metric_list)

        # model, optimizer, scheduler, history, test_result, metric_list = experiment.fit()
        # save_result(args, model, optimizer, scheduler, history, test_result, args.final_epoch, metric_list=metric_list)

def get_scores(y_true, y_pred, score_fun):
    nclasses = np.shape(y_true)[1]
    scores = []
    for name, fun in score_fun.items():
        scores += [[fun(y_true[:, k], y_pred[:, k]) for k in range(nclasses)]]
    return np.array(scores).T

def specificity_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spc = m[0, 0] * 1.0 / (m[0, 0] + m[0, 1])
    return spc

if __name__=='__main__' :
    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself!')
    parser.add_argument('--data_path', type=str, default='/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/dataset/SC1D_dataset')
    parser.add_argument('--data_type', type=str, required=True, choices=['12ECG', '12ECGPaper'])
    parser.add_argument('--model_name', type=str, required=True, choices=['RNN',
                                                                          'LSTM', 'Bi-LSTM',
                                                                          'ResNet1D_18', 'ResNet1D_34', 'ResNet1D_50', 'ResNet1D_101',
                                                                          'Modified_ResNet1D',
                                                                          'MultiResNet1D',
                                                                          'VGG2D_11'])
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    # /media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/model_weights/IS2D_model_weights
    parser.add_argument('--save_path', type=str, default='/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/model_weights/SC1D_model_weights')
    parser.add_argument('--save_cpt_interval', type=int, default=None)
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--reproducibility', default=False, action='store_true')

    # Multi-Processing parameters
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')

    # Train parameter
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--criterion', type=str, default='MLSM', choices=['MLSM',
                                                                          'BCELoss', 'BCEWithLogitsLoss'])
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--final_epoch', type=int, default=200)

    # Optimizer Configuration
    parser.add_argument('--optimizer_name', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # Learning Rate Scheduler (LRS) Configuration
    parser.add_argument('--LRS_name', type=str, default=None)

    # Print parameter
    parser.add_argument('--step', type=int, default=10)

    args = parser.parse_args()

    SC1D_main(args)