import torch.nn.functional as F
import numpy as np
from options.train_options import TrainOptions
from utils.timer import Timer
import os
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from model import CreateModel
#import tensorboardX
import torch.backends.cudnn as cudnn
import torch
from torch.autograd import Variable
#from utils import FDA_source_to_target
import scipy.io as sio
from tqdm.notebook import tqdm

IMG_MEAN_trg = np.array((34.91212110, 145.54035585, 168.38381212), dtype=np.float32)
#IMG_MEAN_trg = np.array((27.69365370, 153.46831124, 174.91789185), dtype=np.float32)
IMG_MEAN_src = np.array((9.8295929, 59.56474619, 75.43405386), dtype=np.float32)

IMG_MEAN_trg = torch.reshape( torch.from_numpy(IMG_MEAN_trg), (1,3,1,1)  )
IMG_MEAN_src = torch.reshape( torch.from_numpy(IMG_MEAN_src), (1,3,1,1)  )
#CS_weights = np.array( (1, 1, 1.1166), dtype=np.float32 )
#CS_weights = np.array( (1, 1.5,1.5), dtype=np.float32 )
CS_weights = np.array( (1, 1, 1), dtype=np.float32 )
CS_weights = torch.from_numpy(CS_weights)


def main():
    opt = TrainOptions()
    args = opt.initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    _t = {'iter time' : Timer()}

    model_name = args.source + '_to_' + args.target
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    opt.print_options(args)

    sourceloader, targetloader = CreateSrcDataLoader(args), CreateTrgDataLoader(args)
    sourceloader_iter, targetloader_iter = iter(sourceloader), iter(targetloader) ## iter: Get each image(stored in tensor form) in DataLodaer

    model, optimizer = CreateModel(args)

    start_iter = 0
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])

    cudnn.enabled = True
    cudnn.benchmark = True

    model.train() ## Put model in train model(including Dropout and Batch Normalization)
    model.cuda()

    # losses to log
    loss = ['loss_seg_src', 'loss_seg_trg']
    loss_train = 0.0
    loss_val = 0.0
    loss_train_list = []
    loss_val_list = []

    mean_img = torch.zeros(1, 1)
    class_weights = Variable(CS_weights).cuda() ## torch.autograd.Variable: Pack data and gradient into "Variable" to translate to cuda

    _t['iter time'].tic()
    for i in range(start_iter, args.num_steps):
        model.adjust_learning_rate(args, optimizer, i)                               # adjust learning rate
        optimizer.zero_grad()                                                        # zero grad

        src_img, src_lbl, _, _ = next(sourceloader_iter)                           # new batch source
        trg_img, trg_lbl, _, _ = next(targetloader_iter)                           # new batch target
        
        scr_img_copy = src_img.clone() ## .clone(): total deep copy

        if mean_img.shape[-1] < 2:
            B, C, H, W = src_img.shape
            mean_img_src = IMG_MEAN_src.repeat(B,1,H,W)
            mean_img_trg = IMG_MEAN_trg.repeat(B,1,H,W)

        #-------------------------------------------------------------------#
        # subtract mean
        ##print('mean_img:{}'.format(mean_img.shape))
        src_img = src_img.clone() - mean_img_src                                 
        trg_img = trg_img.clone() - mean_img_trg 

        #src_domain_lbl = torch.zeros(B, H, W)
        #trg_domain_lbl = torch.ones(B, H, W)                                
        src_domain_lbl = torch.zeros(B)
        trg_domain_lbl = torch.ones(B)                                


        #-------------------------------------------------------------------#
        # evaluate and update params #####
        src_img, src_lbl, src_domain_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda(), Variable(src_domain_lbl.long()).cuda() # to gpu
        alpha = (2/(1+np.exp(-10 * i))) - 1
        #alpha = 1
        #alpha = 0.1
        #alpha = 10
        #print('src_img: {}, src_lbl: {}'.format(src_img.shape, src_lbl.shape))
        src_seg_score = model(src_img, alpha, seg_lbl=src_lbl, domain_lbl=src_domain_lbl, weight=class_weights)      # forward pass
        #src_seg_score = model(src_img, alpha=None, seg_lbl=src_lbl, domain_lbl=None, weight=class_weights)      # forward pass
        
        loss_seg_src = model.loss_seg                                                # get loss
        loss_domain_src = model.domain_loss
        
        
        # get target loss
        trg_img, trg_lbl, trg_domain_lbl = Variable(trg_img).cuda(), Variable(trg_lbl.long()).cuda(), Variable(trg_domain_lbl.long()).cuda() # to gpu
        trg_seg_score = model(trg_img, alpha, seg_lbl=trg_lbl, domain_lbl=trg_domain_lbl, weight=class_weights)      # forward pass
        loss_seg_trg = model.loss_seg
        loss_domain_trg = model.domain_loss
        
        loss_domain = loss_domain_src + loss_domain_trg
        loss_all = loss_seg_src + (loss_domain_src + loss_domain_trg)     
        #loss_all = loss_seg_src
        
        loss_all.backward()
        optimizer.step()

        loss_train += loss_seg_src.detach().cpu().numpy() ## .detach():Return a "Variable" which will not be updated
        loss_val   += loss_seg_trg.detach().cpu().numpy()
        
        
            
        if (i+1) % args.print_freq == 0:  ## Show loss every "print_freq" times
            _t['iter time'].toc(average=False)   ## Return traning time, time is a def function

            
            print('[it %d][src seg loss %.4f][domain loss %.4f][loss_all %.4f][loss_val %.4f][lr %.4f][%.2fs]' % \
                    (i + 1, loss_seg_src.data, loss_domain.data, loss_all.data, loss_seg_trg.data, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff) )

            if (i+1) >= 22000  and loss_seg_src.data <=0.05: 
            #if (i+1) >= 1  :
                print('taking snapshot in process ...')
                torch.save( model.state_dict(), os.path.join(args.snapshot_dir, 'new' + str(i+1) + '.pth') )  
                
            '''          
            print('[it %d][src seg loss %.4f][lr %.4f][%.2fs]' % \
                    (i + 1, loss_seg_src.data, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff) )
   
            if (i+1) >= 4000 and loss_seg_src.data <=0.04: 
                print('taking snapshot in process ...')
                torch.save( model.state_dict(), os.path.join(args.snapshot_dir, '%s_' % (args.source) + str(i+1) + '.pth') )
            '''
         
            sio.savemat(args.tempdata, {'src_img':src_img.cpu().numpy(), 'trg_img':trg_img.cpu().numpy()})

            loss_train /= args.print_freq
            loss_val   /= args.print_freq
            loss_train_list.append(loss_train)
            loss_val_list.append(loss_val)
            sio.savemat( args.matname, {'loss_train':loss_train_list, 'loss_val':loss_val_list} )
            #sio.savemat( args.matname, {'loss_train':loss_train_list} )
            loss_train = 0.0
            loss_val = 0.0

            if i + 1 > args.num_steps_stop:
                print('finish training')
                break
            _t['iter time'].tic()

if __name__ == '__main__':
    main()

