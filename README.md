# NBDNet   
The trained model and testing code of the paper:   
Li H, Wang J, Ai C, Wu Y, Ren X. NBDNet: A Self-Supervised CNN-Based Method for InSAR Phase and Coherence Estimation. Remote Sensing. 2025; 17(7):1181. https://doi.org/10.3390/rs17071181  
   
If you use this code, please cite our paper:     
bibtex   
@Article{rs17071181,   
AUTHOR = {Li, Hongxiang and Wang, Jili and Ai, Chenguang and Wu, Yulun and Ren, Xiaoyuan},   
TITLE = {NBDNet: A Self-Supervised CNN-Based Method for InSAR Phase and Coherence Estimation},   
JOURNAL = {Remote Sensing},   
VOLUME = {17},   
YEAR = {2025},   
NUMBER = {7},   
ARTICLE-NUMBER = {1181},   
URL = {https://www.mdpi.com/2072-4292/17/7/1181},   
ISSN = {2072-4292},   
DOI = {10.3390/rs17071181}   
}   
   
For any comment, suggestion or question, please contact Hongxiang Li (lihongxiang16@mails.ucas.ac.cn).   
   
# Features     
- Capable of suppressing the phase noise in the interferograms while preserving the detail information;   
- Capable to estimating the interferometric phase and coherence simultaneously;   
- Capable to performing denoising in both single-look and multi-look cases.   
   
# Dependence   
Python 3.x     
tensorflow 2.x   
keras 2.x   
numpy   
matplotlib (for visualization)   
   
# Usage    
"data" folder contains a pair of coregistered single look complex (SLC) SAR images (the flat earth phase has been removed from the slave SLC);   
"results" folder contains the denoised interferograms by NBDNet;   
"NBDNet_model.hdf5" is the model file of NBDNet;   
"NBDNet.py" is the model implementation script;   
"demo_single_look.py" is the demo script for single-look denoising;   
"demo_multi_look.py" is the demo script for multi-look denoising.   

