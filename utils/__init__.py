import torch
import numpy as np

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    #fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    #fft_amp = torch.sqrt(fft_amp)
    #fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    fft_amp = fft_im.real**2 + fft_im.imag**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im.imag, fft_im.real)
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) ) # Put low frequency into the certer position
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int) # np.floor: dowm Rounding, np.amin:find min value 
    c_h = np.floor(h/2.0).astype(int) # astype: change data type
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    #a_src[:,0:h1,0:w1] = a_trg[:,0:h1,0:w1]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    ##fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    ##fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )
    fft_src = torch.fft.fft2( src_img.clone(), dim=(-2, -1)) 
    fft_trg = torch.fft.fft2( trg_img.clone(), dim=(-2, -1))
    #print('fft_src:{}'.format(fft_src.size()))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())
    
    ## Divide tensor in two dimenson in terms of real and imagine value
    fft_src_dim2 = torch.stack((fft_src.real, fft_src.imag),-1)
    
    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src_dim2.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()
    #fft_src_real = torch.zeros( fft_src.size(), dtype=torch.float )
    #fft_src_imag = torch.zeros( fft_src.size(), dtype=torch.float )
    #fft_src_real = torch.cos(pha_src.clone()) * amp_src_.clone()
    #fft_src_imag = torch.sin(pha_src.clone()) * amp_src_.clone()

    fft_src_ = torch.complex(fft_src_[:,:,:,:,0], fft_src_[:,:,:,:,1]) 
    ##print('fft_src_complex:{}'.format(fft_src_.size()))

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    #src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )
    src_in_trg = torch.fft.irfft2(fft_src_, s=(imgH, imgW))
    ##print('Re_fft_src:{}'.format(src_in_trg.size()))
    
    return src_in_trg

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) ) # Do Fast-Fourier-Transform along last two axis
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) ) # Output is complex ndarray

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np) # Get abs and theta value
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg

