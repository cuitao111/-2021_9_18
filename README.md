# 问题：
## 将代码从本地服务器迁移到集群上时发生如下错误
```
(alphapose) [u20120297@gpu01 AlphaPose]$ python video_inference.py 
Traceback (most recent call last):
  File "video_inference.py", line 17, in <module>
    from alphapose.models import builder
  File "/data/home/u20120297/code/AlphaPose/alphapose/models/__init__.py", line 2, in <module>
    from .fastpose_duc import FastPose_DUC
  File "/data/home/u20120297/code/AlphaPose/alphapose/models/fastpose_duc.py", line 11, in <module>
    from .layers.ShuffleResnet import ShuffleResnet
  File "/data/home/u20120297/code/AlphaPose/alphapose/models/layers/ShuffleResnet.py", line 9, in <module>
    from .dcn import DCN
  File "/data/home/u20120297/code/AlphaPose/alphapose/models/layers/dcn/__init__.py", line 1, in <module>
    from .deform_conv import (DeformConv, DeformConvPack, ModulatedDeformConv,
  File "/data/home/u20120297/code/AlphaPose/alphapose/models/layers/dcn/deform_conv.py", line 9, in <module>
    from . import deform_conv_cuda
ImportError: /data/home/u20120297/code/AlphaPose/alphapose/models/layers/dcn/deform_conv_cuda.cpython-36m-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC1ENS_14SourceLocationERKSs
```
# 解决步骤
搜索问题，网上给出的解答是编译环境和运行环境不同导致的，没太理解，对比了本地服务器和集群的conda环境
### 本地服务器
pytorch==1.1.0
torchvision==0.3.0
cudatoolkit=10.0
gcc==7.5.0
### 集群
pytorch==1.1.0
torchvision==0.3.0
cudatoolkit=10.1
gcc==7.5.0

cuda版本不一致，集群的/usr/local下刚好有cuda-10.0,以为问题解决了

因为集群没有root权限，直接修改~/.bashrc
```
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0
exportPATH=$PATH:/usr/local/cuda-10.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
source ~/.bashrc
```
运行还是报错，之后看了代码的配置环境文档
```
# 1. Create a conda virtual environment.
conda create -n alphapose python=3.6 -y
conda activate alphapose

# 2. Install PyTorch
conda install pytorch==1.1.0 torchvision==0.3.0

# 3. Get AlphaPose
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose

# 4. install
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
python -m pip install cython
sudo apt-get install libyaml-dev

python setup.py build develop
```
觉得问题应该在最后setup一步，但是不想深入看这个文件，于是又google了这个相似的[问题](https://github.com/MVIG-SJTU/AlphaPose/issues/583)
说是把gcc换成<6的版本，于是换成了gcc-5.4,还是报错。

PS:[非root用户编译安装切换gcc](https://blog.csdn.net/happyeveryday62/article/details/107673477)


# 最终解决方法
看了下setup.py的内容，找到其中有关deform_conv_cuda.cpython-36m-x86_64-linux-gnu.so的代码
```
setup(
        ...
        ext_modules=get_ext_modules(),
        ...
        )
def get_ext_modules():
    ...
    if platform.system() != 'Windows' or force_compile:
        ext_modules = [
            ....
            make_cuda_ext(
                name='deform_conv_cuda', 
                module='alphapose.models.layers.dcn',
                sources=[
                     'src/deform_conv_cuda.cpp',
                     'src/deform_conv_cuda_kernel.cu'
        ]
    return ext_modules
```
另外，执行setup.py的时候有下面几句输出
```
creating build
    ...
creating build/temp.linux-x86_64-3.6/alphapose/models/layers/dcn/src
gcc -pthread -B /data/home/u20120297/.conda/envs/alphapose/compiler_compat -Wl,--sysroot=/...
    ...
copying build/lib.linux-x86_64-3.6/alphapose/models/layers/dcn/deform_conv_cuda.cpython-36m-x86_64-linux-gnu.so -> alphapose/models/layers/dcn

```

应该就是把src下的几个文件给编译成deform_conv_cuda.cpython-36m-x86_64-linux-gnu.so,然后把编译的.o文件移动到代码目录下面

去gibhub代码地址看了下，src是作者提供的应该没问题，那问题就是build文件夹了，进入目录下查看

```
(alphapose) [u20120297@gpu01 build]$ ls
lib.linux-x86_64-3.6  lib.linux-x86_64-3.8  temp.linux-x86_64-3.6  temp.linux-x86_64-3.8
```
当时复制代码的时候把这个build以及alphapose.egg-info egg包一起复制了，运行setup的时候检测到有lib.linux-x86_64-3.6和temp.linux-x86_64-3.6就没再编译一遍，也就是说，build里面的.o文件是本地服务器环境下编译的，而代码运行的环境是集群的，终于理解上面说的编译环境和运行环境不同是啥意思了，꒰*´◒`*꒱
后面把build和alphapose.egg-info删掉之后重新setup一遍就好了

# 后续

唯一不理解的是我把集群的环境改成和本地服务器相同的环境之后，编译环境和运行环境就相同了，但依然会报错，可能两个服务器环境的某些地方还是不一致把~

另外，当时是对源代码做了修改，为了图方便就直接copy了整个文件夹到集群才踩到了坑，如果先git clone下再修改代码，或者只复制py文件就不会有问题。
