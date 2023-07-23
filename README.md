## 基于MindSpore框架复现Unet-AG的论文

##### 参加的是【MindSpore开发者群英会】经典论文复现活动，选择了第16个任务进行在原有的基础上进行模型的复现。

##### 包括训练和推理都可以实现

原论文地址：https://paperswithcode.com/paper/attention-u-net-learning-where-to-look-for

MindSpore开发者群英会：https://gitee.com/mindspore/community/issues/I6Q8R0

Csdn：https://blog.csdn.net/professor006/article/details/131874950?spm=1001.2014.3001.5502

数据使用的是细胞分割的数据集进行训练的推理，我的dataset直接有数据集，可以直接进行使用

使用的环境是mindspore2.0版本  Ascend

使用的是unet_medical_config.yaml 文件，运行的时候只需修改数据输入的路径即可

注意：如果想使用unet_AG网络，在train.py下更改

```
from src.unet_AG import UNetMedical
```

想使用原本的unet，在train.py下更改

```
from src.unet_medical import UNetMedical
```

训练的时候运行

```
python train.py --data_path=dataset/ --config_path=unet_medical_config.yaml 

```

推理的时候运行

```
python eval.py --data_path=dataset/ --config_path=unet_medical_config.yaml 
```

（注意在不同操作环境下，可能需要更改一下路径）
