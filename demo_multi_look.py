# -*- coding: utf-8 -*-
"""
NBDNet denoising for multi-look interferogram

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
    
    # Generate multi-look interferogram
    look_y, look_x = 4, 4
    noisy_intf = NBDNet.slc_intf(slc1, slc2, look_y, look_x)
    
    # NBDNet denoising
    print('NBDNet denoising for multi-look interferogram...')
    start_time = time.time()
    intf_denoised = NBDNet.denoise_multi(noisy_intf)
    elapsed_time = time.time() - start_time   
    print('Elapsed time: %.1fs' % elapsed_time)

    # save the result
    np.save(os.path.join('results', 'multi_look_denoised.npy'), intf_denoised)
    
    # Display the result
    plt.figure(figsize=(10, 10),dpi = 600)
    plt.subplot(221)
    plt.imshow(np.angle(noisy_intf),vmin=-np.pi,vmax=np.pi,cmap='jet'), plt.colorbar(shrink=0.82), plt.title('Noisy Phase')
    plt.subplot(222)
    plt.imshow(np.abs(noisy_intf),vmin=0,vmax=1,cmap='jet'), plt.colorbar(shrink=0.82), plt.title('Noisy Coherence')
    plt.subplot(223)
    plt.imshow(np.angle(intf_denoised),vmin=-np.pi,vmax=np.pi,cmap='jet'), plt.colorbar(shrink=0.82), plt.title('Denoised Phase')
    plt.subplot(224)
    plt.imshow(np.abs(intf_denoised),vmin=0,vmax=1,cmap='jet'), plt.colorbar(shrink=0.82), plt.title('Estimated Coherence')  
    plt.show()

    
    