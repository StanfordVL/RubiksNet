# RubiksNet: Learnable 3D-Shift for Efficient Video Action Recognition

This repository contains the official PyTorch implementation with 
accelerated CUDA kernels for our paper:

> [RubiksNet: Learnable 3D-Shift for Efficient Video Action Recognition](https://rubiksnet.stanford.edu/)<br/>
> <b>ECCV 2020</b><br/>
> <b>Linxi (Jim) Fan*, Shyamal Buch*, Guanzhi Wang, Ryan Cao, Yuke Zhu, Juan Carlos Niebles, Li Fei-Fei</b><br/>
> (* denotes equal contribution lead author)

<b>Quick Links:</b>
[[paper](https://stanfordvl.github.io/rubiksnet-site//assets/eccv20.pdf)]
[[project website](https://rubiksnet.stanford.edu/)]
[[video](https://youtu.be/3alaXltwEWw)]
[[eccv page](https://papers.eccv2020.eu/paper/3271/)]
[[supplementary](https://stanfordvl.github.io/rubiksnet-site//assets/eccv20_supplement.pdf)]
[[code](https://github.com/StanfordVL/rubiksnet)]

## Abstract

![framework](https://stanfordvl.github.io/rubiksnet-site//assets/images/pullfig-nolabels.png)

Video action recognition is a complex task dependent on modeling spatial and temporal context. Standard approaches rely on 2D or 3D convolutions to process such context, resulting in expensive operations with millions of parameters. Recent efficient architectures leverage a channel-wise shift-based primitive as a replacement for temporal convolutions, but remain bottlenecked by spatial convolution operations to maintain strong accuracy and a fixed-shift scheme. Naively extending such developments to a 3D setting is a difficult, intractable goal.

To this end, we introduce <b>RubiksNet</b>, a new efficient architecture for video action recognition based on a proposed learnable 3D spatiotemporal shift operation (<b>RubiksShift</b>). We analyze the suitability of our new primitive for video action recognition and explore several novel variations of our approach to enable stronger representational flexibility while maintaining an efficient design. We benchmark our approach on several standard video recognition datasets, and observe that our method achieves comparable or better accuracy than prior work on efficient video action recognition at a fraction of the performance cost, with <b>2.9 - 5.9x fewer parameters</b> and <b>2.1 - 3.7x fewer FLOPs</b>. We also perform a series of controlled ablation studies to verify our significant boost in the efficiency-accuracy tradeoff curve is rooted in the core contributions of our RubiksNet architecture.

## Installation

Tested with:

* Ubuntu 18.04
* PyTorch >= 1.5
* CUDA 10.1

```
# Create your virtual environment
conda create --name rubiksnet python=3.7
conda activate rubiksnet

# Install PyTorch and supporting libraries
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install scikit-learn

# Clone this repo
git clone https://github.com/stanfordvl/rubiksnet.git
cd rubiksnet

# Compiles our efficient CUDA-based RubiksShift operator under the hood
# and installs the main API
pip install -e .
```

To test if the installation is successful, please run `python scripts/test_installation.py`. 
You should see a random prediction followed by "Installation successful!".

## Usage

It is very simple to get started with our API: 

```python
from rubiksnet.models import RubiksNet

# `tier` must be one of ["tiny", "small", "medium", "large"]
# `variant` must be one of ["rubiks3d", "rubiks3d-aq"]
# 174 is the number of classes for Something-Something-V2 action classification

# instantiate RubiksNet-Tiny with random weights
net = RubiksNet(tier="tiny", num_classes=174, variant="rubiks3d")

# instantiate RubiksNet-Large network with temporal attention quantized shift 
net = RubiksNet(tier="large", num_classes=174, variant="rubiks3d-aq")

# load RubiksNet-Large model from pretrained weights
net = RubiksNet.load_pretrained("pretrained/ssv2_large.pth.tar")
```

From here, `net` contains a RubiksNet model and can be used like any other PyTorch model! See our [inference script](scripts/test_models.py) for example usage.

## Pretrained Models

### Something-Something-V2

For the [Something-Something-V2 benchmark](https://20bn.com/datasets/something-something), we follow the evaluation convention in [TSM](https://github.com/mit-han-lab/temporal-shift-module) and report results from two evaluation protocols. For "1-Clip Val Acc", we sample only a single clip per video and the center 224×224 crop for evaluation. For "2-Clip Val Acc", we sample 2 clips per video and take 3 equally spaced 224×224 crops from the full resolution image scaled to 256 pixels on the shorter side.

| Model                          | Input  | 2-Clip Top-1 | 2-Clip Top-5 | #Param. | FLOPs | Test Log | Pretrained |
| ------------------------------ | ------ | ------------ | -----------  | ------- |       ----- | -------- | ---------- | 
| RubiksNet-<br>Large-<br>AQ (Budget=0.125)      | 8      | 61.6         | 86.7         | 8.5M    | 15.7G       | [1-clip](scripts/eval_logs/ssv2_large_aq_1clip.log)<br>[2-clip](scripts/eval_logs/ssv2_large_aq_2clip.log) | [model link](pretrained/ssv2_large_aq_budget0.125.pth.tar)|
| RubiksNet-<br>Large                | 8      | 61.7         | 87.3         | 8.5M    | 15.8G       | [1-clip](scripts/eval_logs/ssv2_large_1clip.log)<br>[2-clip](scripts/eval_logs/ssv2_large_2clip.log) | [model link](pretrained/ssv2_large.pth.tar) |
| RubiksNet-<br>Medium               | 8      | 60.8         | 86.9         | 6.2M    | 11.2G       | [1-clip](scripts/eval_logs/ssv2_medium_1clip.log)<br>[2-clip](scripts/eval_logs/ssv2_medium_2clip.log) | [model link](pretrained/ssv2_medium.pth.tar) |
| RubiksNet-<br>Small                | 8      | 59.8         | 86.2         | 3.6M    | 6.8G        | [1-clip](scripts/eval_logs/ssv2_small_1clip.log)<br>[2-clip](scripts/eval_logs/ssv2_small_2clip.log) | [model link](pretrained/ssv2_small.pth.tar) |
| RubiksNet-<br>Tiny                 | 8      | 56.7         | 84.1         | 1.9M    | 3.9G        | [1-clip](scripts/eval_logs/ssv2_tiny_1clip.log)<br>[2-clip](scripts/eval_logs/ssv2_tiny_2clip.log) | [model link](pretrained/ssv2_tiny.pth.tar) |

### Kinetics

We also provide pretrained models on the [Kinetics dataset](https://arxiv.org/abs/1705.07750), which follow the *pretraining* protocol in [prior work](https://github.com/mit-han-lab/temporal-shift-module) -- see the supplementary material for details. All four tiers of RubiksNet can be found at `pretrained/kinetics_{large,medium,small,tiny}.pth.tar`. 

Our CUDA implementation includes accelerated gradient calculation on GPUs. 
We provide an example script to finetune the pretrained kinetics checkpoints on your own dataset. 

```bash
python scripts/example_finetune.py --gpu 0 --pretrained-path pretrained/kinetics_tiny.pth.tar
``` 

The script contains a dummy dataset that generates random videos. You can replace 
it with your own data loader. If you run the above script, you should see RubiksNet-tiny 
gradually overfitting the artificial training data. 


## Testing 

Please refer to [TSM](https://github.com/yjxiong/temporal-segment-networks) repo for how to prepare the Something-Something-V2 test data. We assume the processed dataset is located at `<root_path>/somethingv2`

To test "2-Clip Val Acc" of the pretrained SomethingV2 models, you can run

```bash
# test RubiksNet-Large
python test_models.py somethingv2 
	--root-path=<root_path_to_somethingv2_dataset> \
	--pretrained=pretrained/ssv2_large.pth.tar \
	--two-clips \
	--batch-size=80 -j 8 
```

To test "1-Clip Val Acc", you can run

```bash
# test RubiksNet-Large
python test_models.py somethingv2 \
	--root-path=<root_path_to_somethingv2_dataset> \
	--pretrained=pretrained/ssv2_large.pth.tar \
	--batch-size=80 -j 8 
```


## Citation

If you find this code useful, please cite our ECCV paper:


```
@inproceedings{fanbuch2020rubiks,
  title={RubiksNet: Learnable 3D-Shift for Efficient Video Action Recognition},
  author={Linxi Fan* and Shyamal Buch* and Guanzhi Wang and Ryan Cao and Yuke Zhu and Juan Carlos Niebles and Li Fei-Fei},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

### LICENSE

We release our code here under the open [MIT License](LICENSE). Our contact information can be found in the paper and on our project website.

### Acknowledgements
This research was sponsored in part by grants from Toyota Research Institute (TRI). Some computational support for experiments was provided by Google Cloud and NVIDIA. The authors also acknowledge fellowship support. Please refer to our paper for full acknowledgements, thank you!

We reference code from the excellent repos of
[Temporal Segment Network](https://github.com/yjxiong/temporal-segment-networks),
[Temporal Shift Module](https://github.com/mit-han-lab/temporal-shift-module),
[ShiftResNet](https://github.com/alvinwan/shiftresnet-cifar), and
[ActiveShift](https://github.com/jyh2986/Active-Shift).
Please be sure to cite these works/repos as well.
