Learning Dual Encoding Model for Adaptive Visual Understanding in Visual Dialogue
====================================

![alt text](image/visual_result.png)
<p align="center">Example results from the VisDial v1.0 validation dataset.</p>



This is a PyTorch implementation for [Learning Dual Encoding Model for Adaptive Visual Understanding in Visual Dialogue, IEEE Transactions on Image Processing](https://ieeexplore.ieee.org/document/9247486).


  * [Requirements](#Requirements)
  * [Data](#Data)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Acknowledgements](#acknowledgements)

If you use this code in your research, please consider citing:

```text
@article{yu2020learning,
  title =  {Learning Dual Encoding Model for Adaptive Visual Understanding in Visual Dialogue},
  author =  {Yu, Jing and Jiang, Xiaoze and Qin, Zengchang and Zhang, Weifeng and Hu, Yue and Wu, Qi},
  journal = {IEEE Transactions on Image Processing},
  volume={30},
  pages={220--233},
  year =  {2020}
}
```

A previous version of our dual encoding model was published in AAAI 2020: DualVD: An Adaptive Dual Encoding Model for Deep Visual Understanding in Visual Dialogue. [[Paper]](https://arxiv.org/abs/1911.07251) [[Code]](https://github.com/JXZe/DualVD)


Requirements
----------------------
This code is implemented using PyTorch v1.0, and provides out of the box support with CUDA 9 and CuDNN 7. 

```sh
conda create -n visdialch python=3.6
conda activate visdialch  # activate the environment and install all dependencies
cd DualVD/
pip install -r requirements.txt
```




Data
----------------------

1. Download the VisDial v1.0 dialog json files and images from [here][1].
2. Download the word counts file for VisDial v1.0 train split from [here][2]. 
They are used to build the vocabulary.
3. Use Faster-RCNN to extract image features from [here][3].
4. Use Large-Scale-VRD to extract visual relation embedding from [here][4].
5. Use Densecap to extract local captions from [here][5].
6. Generate ELMo word vectors from [here][6].
7. Download pre-trained GloVe word vectors from [here][7].


Training
--------


Train the DualVD model as:

```sh
python train.py --config-yml configs/lf_disc_faster_rcnn_x101_bs32.yml --gpu-ids 0 1 # provide more ids for multi-GPU execution other args...
```

The code have an `--overfit` flag, which can be useful for rapid debugging. It takes a batch of 5 examples and overfits the model on them.

##### DualVD-LF 
DualVD-LF is trained with [configs/lf_disc_faster_rcnn_x101.yml](https://github.com/JXZe/Learning_DualVD/blob/main/configs/lf_disc_faster_rcnn_x101.yml).

##### DualVD-MN
DualVD-MN is trained with [configs/mn_disc_faster_rcnn_x101.yml](https://github.com/JXZe/Learning_DualVD/blob/main/configs/mn_disc_faster_rcnn_x101.yml).


### Saving model checkpoints

This script will save model checkpoints at every epoch as per path specified by `--save-dirpath`. Refer [visdialch/utils/checkpointing.py][8] for more details on how checkpointing is managed.

### Logging

Use [Tensorboard][9] for logging training progress. Recommended: execute `tensorboard --logdir /path/to/save_dir --port 8008` and visit `localhost:8008` in the browser.


Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:

```sh
python evaluate.py --config-yml /path/to/config.yml --load-pthpath /path/to/checkpoint.pth --split val --gpu-ids 0
```







Acknowledgements
----------------

* This code began with [batra-mlp-lab/visdial-challenge-starter-pytorch][10]. We thank the developers for doing most of the heavy-lifting.


[1]: https://visualdialog.org/data
[2]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json
[3]: https://github.com/peteanderson80/bottom-up-attention
[4]: https://github.com/jz462/Large-Scale-VRD.pytorch
[5]: https://github.com/jcjohnson/densecap
[6]: https://allennlp.org/elmo
[7]: https://github.com/stanfordnlp/GloVe
[8]: https://github.com/JXZe/DualVD/blob/master/visdialch/utils/checkpointing.py
[9]: https://www.github.com/lanpa/tensorboardX
[10]: https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch

