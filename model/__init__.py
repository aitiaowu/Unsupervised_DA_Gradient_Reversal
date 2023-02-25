from model.our_model import ResNet
import torch.optim as optim


def CreateModel(args):
    if args.model == 'our_model':
        phase = 'test'
        if args.set == 'train' or args.set == 'trainval':
            phase = 'train'
        model = ResNet(num_classes=args.num_classes, init_weights=args.init_weights, restore_from=args.restore_from, phase=phase)
        print('num_classes = ', args.num_classes)
        if args.set == 'train' or args.set == 'trainval':
            optimizer = optim.SGD(model.optim_parameters(args),
                                  lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            optimizer.zero_grad()
            return model, optimizer
        else:
            return model


