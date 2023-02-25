import numpy as np
from torch.utils import data
from data.young_dataset import youngDataSet
from data.old_dataset import oldDataSet
from data.old_dataset_label import oldDataSetLabel


IMG_MEAN = np.array((0.0, 0.0, 0.0), dtype=np.float32)
image_sizes = {'old': (672,376), 'young': (672,376)}
AS_size_test = {'old': (672,376)}

def CreateSrcDataLoader(args):
    if args.source == 'old':
        source_dataset = youngDataSet( args.data_dir, args.data_list, crop_size=image_sizes['old'], 
                                      resize=image_sizes['old'] ,mean=IMG_MEAN,
                                      max_iters=args.num_steps * args.batch_size )
    else:
        raise ValueError('The source dataset mush be either Blender or synthia')
    
    source_dataloader = data.DataLoader( source_dataset, 
                                         batch_size=args.batch_size,
                                         shuffle=True, 
                                         num_workers=args.num_workers, 
                                         pin_memory=True )    
    return source_dataloader

def CreateTrgDataLoader(args):
    if args.set == 'train' or args.set == 'trainval':
        target_dataset = oldDataSetLabel( args.data_dir_target, 
                                                 args.data_list_target, 
                                                 crop_size=image_sizes['old'], 
                                                 mean=IMG_MEAN, 
                                                 max_iters=args.num_steps * args.batch_size, 
                                                 set=args.set )
    else:
        target_dataset = oldDataSet( args.data_dir_target,
                                            args.data_list_target,
                                            crop_size=AS_size_test['old'],
                                            mean=IMG_MEAN,
                                            set=args.set )

    if args.set == 'train' or args.set == 'trainval':
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True )
    else:
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=1, 
                                             shuffle=False, 
                                             pin_memory=True )

    return target_dataloader




