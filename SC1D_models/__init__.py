import timm

from .rnn import *

def SC1D_model(args) :
    if args.model_name == 'RNN' :
        from SC1D_models.rnn import RNN
        return RNN(input_dim=args.num_channels, hidden_dim=64, num_classes=args.num_classes, dropout_prob=0.5)
    elif args.model_name == 'LSTM':
        from SC1D_models.lstm import LSTM
        return LSTM(input_size=args.num_channels, hidden_size=64, seq_length=args.seq_length, num_classes=args.num_classes, bidirectional=False)
    elif args.model_name == 'Bi-LSTM':
        from SC1D_models.lstm import LSTM
        return LSTM(input_size=args.num_channels, hidden_size=64, seq_length=args.seq_length, num_classes=args.num_classes, bidirectional=True)
    elif args.model_name == 'ResNet1D_18':
        from SC1D_models.resnet1d import resnet1d_18
        return resnet1d_18(input_dim=args.num_channels, num_classes=args.num_classes)
    elif args.model_name == 'ResNet1D_34':
        from SC1D_models.resnet1d import resnet1d_34
        return resnet1d_34(input_dim=args.num_channels, num_classes=args.num_classes)
    elif args.model_name == 'ResNet1D_50':
        from SC1D_models.resnet1d import resnet1d_50
        return resnet1d_50(input_dim=args.num_channels, num_classes=args.num_classes)
    elif args.model_name == 'ResNet1D_101':
        from SC1D_models.resnet1d import resnet1d_101
        return resnet1d_101(input_dim=args.num_channels, num_classes=args.num_classes)
    elif args.model_name == 'Modified_ResNet1D' :
        from SC1D_models.resblk1d import ResNet1d
        return ResNet1d(input_dim=(args.num_channels, args.seq_length),
                        blocks_dim=list(zip([64, 128, 196, 256, 320], [2048, 512, 128, 128, 128])),
                        n_classes=args.num_classes)
    elif args.model_name == 'MultiResNet1D' :
        from SC1D_models.multi_resblk1d import MultiResNet1D
        return MultiResNet1D(input_dim=(args.num_channels, args.seq_length),
                             blocks_dim=list(zip([64, 128, 196, 256, 320], [2048, 512, 128, 128, 128])),
                             n_classes=args.num_classes)
    elif args.model_name == 'VGG2D_11':
        from SC1D_models.vgg import vgg11
        return vgg11(args.image_size, 512, args.num_channels, args.num_classes)