import json
import os.path as osp
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from data import CreateTrgDataLoader
from model import CreateModel
import os
from options.test_options import TestOptions
import scipy.io as sio

# color coding of semantic classes


palette = [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def fast_hist(a, b, n):
    k = (a>=0) & (a<n)
    return np.bincount( n*a[k].astype(int)+b[k], minlength=n**2 ).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / ( hist.sum(1)+hist.sum(0)-np.diag(hist) )

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[ input==mapping[ind][0] ] = mapping[ind][1]
    return np.array(output, dtype=np.int64)
  
def get_img_list(Path):
    img_list = []
    for f in os.listdir(Path):
      if not f.endswith('color.png'):
        img_list.append(f)
    img_list.sort()
    return img_list

def compute_mIoU( gt_dir, pred_dir, num_classes, devkit_dir='', restore_from='' ):

    name_classes = np.array(['Seam', 'Edge', 'Spot'], dtype=np.str)
    #mapping = np.array( info['label2train'],dtype=np.int )
    hist = np.zeros( (num_classes, num_classes) )


    image_path_list = get_img_list(pred_dir)
    label_path_list = get_img_list(gt_dir)

    pred_imgs = [osp.join(pred_dir, x) for x in image_path_list]
    gt_imgs = [osp.join(gt_dir, x) for x in label_path_list]
    
    for ind in range(len(gt_imgs)):
        pred  = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        print('pred size:', pred.shape, 'label size:', label.shape)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format( len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind] ))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            with open(restore_from+'_mIoU.txt', 'a') as f:
                f.write( '{:d} / {:d}: {:0.2f}\n'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))) )
            #print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
 
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        with open(restore_from+'_mIoU.txt', 'a') as f:
            f.write('===>'+name_classes[ind_class]+':\t' + str(round(mIoUs[ind_class]*100,2)) + '\n')
        print('===>'+name_classes[ind_class]+':\t' + str(round(mIoUs[ind_class]*100,2)))
    with open(restore_from+'_mIoU.txt', 'a') as f:
        f.write('===> mIoU: ' + str(round(np.nanmean(mIoUs)*100,2)) + '\n')

    print('===> mIoU: ' + str(round(   np.nanmean(mIoUs)*100,2   )))


def main():
    opt = TestOptions()
    args = opt.initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    args.restore_from = args.restore_opt1
    model1 = CreateModel(args)
    model1.eval()
    model1.cuda()
    '''
    args.restore_from = args.restore_opt2
    model2 = CreateModel(args)
    model2.eval()
    model2.cuda()

    args.restore_from = args.restore_opt3
    model3 = CreateModel(args)
    model3.eval()
    model3.cuda()
    '''

    targetloader = CreateTrgDataLoader(args)

    #IMG_MEAN = np.array((34.91212110, 145.54035585, 168.38381212), dtype=np.float32)
    #IMG_MEAN = np.array((27.69365370, 153.46831124, 174.91789185), dtype=np.float32)
    IMG_MEAN = np.array((9.8295929, 59.56474619, 75.43405386), dtype=np.float32)
    IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
    mean_img = torch.zeros(1, 1)

    # ------------------------------------------------- #
    # compute scores and save them
    with torch.no_grad():
        for index, batch in enumerate(targetloader):
            if index % 2000 == 0:
                print( '%d processd' % index )
            image, _, name = batch                              # 1. get image
            # create mean image
            if mean_img.shape[-1] < 2:
                B, C, H, W = image.shape
                mean_img = IMG_MEAN.repeat(B,1,H,W)             # 2. get mean image
            image = image.clone() - mean_img                    # 3, image - mean_img
            image = Variable(image).cuda()
 
            # forward
            output1 = model1(image)
            output1 = nn.functional.softmax(output1, dim=1)


            #a, b = 0.3333, 0.3333
            #output = a*output1 + b*output2 + (1.0-a-b)*output3
            
            output = output1
            output = nn.functional.interpolate(output, (188,336), mode='bilinear', align_corners=True).cpu().data[0].numpy()
            output = output.transpose(1,2,0)

            output_nomask = np.asarray( np.argmax(output, axis=2), dtype=np.uint8 )
            output_col = colorize_mask(output_nomask) 
            '''
            gt = [osp.join(args.gt_dir, x) for x in get_img_list(args.gt_dir)]
            for ind in range(len(gt)):
            	label = np.array(Image.open(gt[ind]))
            	output_col = colorize_mask(label)
            	output_col.save(  '%s/%s.png' % (args.save, ind)  ) 
            '''
            output_nomask = Image.fromarray(output_nomask)    
            name = name[0].split('/')[-1]
            #output_nomask.save(  '%s/%s' % (args.save, name)  )
            output_col.save(  '%s/%s.png' % (args.save, name.split('.')[0])  ) 
    # scores computed and saved
    # ------------------------------------------------- #
    print('---------Compute IoU------------------')
    compute_mIoU( args.gt_dir, args.save, args.num_classes, args.devkit_dir, args.restore_from )    


if __name__ == '__main__':
    main()

