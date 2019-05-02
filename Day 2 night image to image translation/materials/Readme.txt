We have used the dataset from "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/night2day.tar.gz" and implemented split.py to break the images into night and day instances respectively.

Naive Approach
1. Simply upload all the files on a jupyter notebook online and run each tab of the jupyter notebook
2. To run it on local machine just run the file cv.py it will automatically install the dependencies for you using get_ipython, if that command doesnt work then make sure you have the library of searborn, opencv and pandas installed on your local machine
3. The code is written using python3 so make sure you have the python 3 version of code installed on your local machine
4. Install dependencies using:
pip install searborn
pip install opencv-python

Cycle-Gan Approach 
1. We have used the Pytorch implementation available on github as : "https://github.com/aitorzip/PyTorch-CycleGAN"
2. Download the dataset from 'https://drive.google.com/uc?authuser=0&id=1TEx8yKUEYlDshuIw2ukoGsCSHtRcvBQZ&export=download' and save in the datasets folder.
3. For Training use " python train --dataroot datasets/day2night/ --n_epoch 200 --decay_epoch 1 --cuda --n_cpu 0"
4. For Testing use " python test --dataroot datasets/day2night/ --cuda --n_cpu 0"

Conditional Adversarial Network
1. Run the ipynb file on google collab.
2. To run the file on local machines, follow the same procedure by opening the file in Jupyter Notebook and run the entire code or chunks iteratively.