# -*- coding: utf-8 -*-
"""
NBDNet denoising for single-look interferogram

@author: LiHongxiang
"""

import time, os
import numpy as np
import NBDNet
import matplotlib.pyplot as plt

if __name__ == '__main__': 
    
    # Load coregistered SLCs 
    slc1 = np.load(os.path.join('data', 'slc1.npy'))
    slc2 = np.load(os.path.join('data', 'slc2.npy'))
    
    # NBDNet denoising
    print('NBDNet denoising for single-look interferogram...')
    start_time = time.time()
    intf_denoised = NBDNet.denoise_single(slc1, slc2)
    elapsed_time = time.time() - start_time   
    print('Elapsed time: %.1fs' % elapsed_time)
    
    # save the result
    np.save(os.path.join('results', 'single_look_denoised.npy'), intf_denoised)
        
    # Display the result
    plt.figure(figsize=(30, 10),dpi = 600)
    plt.subplot(131)
    plt.imshow(np.angle(slc1*np.conj(slc2)),vmin=-np.pi,vmax=np.pi,cmap='jet'), plt.colorbar(shrink=0.72), plt.title('Noisy Phase')
    plt.subplot(132)
    plt.imshow(np.angle(intf_denoised),vmin=-np.pi,vmax=np.pi,cmap='jet'), plt.colorbar(shrink=0.72), plt.title('Denoised Phase')
    plt.subplot(133)
    plt.imshow(np.abs(intf_denoised),vmin=0,vmax=1,cmap='jet'), plt.colorbar(shrink=0.72), plt.title('Estimated Coherence')  
    plt.show()
    
