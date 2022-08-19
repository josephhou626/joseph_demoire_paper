# Image Demoireing using Multi-scale Fusion Networks (DMSFN)
[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)]()
[![official_paper](https://img.shields.io/badge/IEEE-Paper-blue)]()


> **Abstract:** 
*Taking images on a digital display may cause a visually-annoying optical effect, called moiré, which degrades image visual quality. In this paper, we propose an Image Demoiréing Multi-scale Fusion network (DMSFN) to remove Moiré patterns and a method for data augmentation using the transfer of Moiré patterns, which can enhance the performance of demoiréing. According to the experimental results, our model performs favorably against state-of-the-art demoiréing methods on benchmark datasets.* 



## Network Architecture of DMSFN

<img src="./Figure/Arch_DMSFN.jpg" width = "800" height = "400" div align=center />

## Environment

- Windows 10 
- GeForce RTX 3090 GPU
- python3.8.6
- torch=1.8.1
- torchvision=0.9.1 

## Installation

1. Install virtual environment:
	```shell
	virtualenv -p python3 exp2 # establish
	.\exp2\Scripts\activate # activate 
	```

2. Clone this repo:
	```shell
	git clone https://github.com/josephhou626/jopseph_thesis.git # clone
	```

3. Install torch and torchvision:
	```shell
	pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
	```

4. Install dependencies:
   ```shell
   pip install -r requirements.txt
   ```
   
## Data preparation
- Download "[TIP18](https://drive.google.com/drive/folders/109cAIZ0ffKLt34P7hOMKUO14j3gww2UC)" dataset into './datasets' </br>
- For example: './datasets/TIP18'

```
./datasets/TIP18
+--- trainData
|   +--- source
|   +--- target
|
+--- TestData
|   +--- source
|   +--- target
```


**Prepossessing  TIP18 Dataset** </br>
- First of all, we will crop the images of the TIP18 dataset to 256*256 resolution to train and test model. 

To generate train data: 
```
python prepossessing.py --data_clear_path datasets/TIP18/trainData/target --data_moire_path datasets/TIP18/trainData/source --save_dir_name TrainData
```
To generate test data:
```
python prepossessing.py --data_clear_path datasets/TIP18/testData/target --data_moire_path datasets/TIP18/testData/source --save_dir_name TestData
```


## Data augmentation for  Moiré patterns
- The implementation is modified from "[Explore Image Deblurring via Encoded Blur Kernel Space](https://github.com/VinAIResearch/blur-kernel-space-exploring)"
- cd './AUG'

## Training for Data augmentation

```
python DA_train.py
```

## Generating for Data augmentation
- For reproducing our results on TIP18 datasets, download "[DA_wight.pth](https://drive.google.com/file/d/1oWk4OHYwtHz52GkGaA8t8fVxIl5Vtv_u/view?usp=sharing)" </br>
- Put the DA_wight.pth into 'experiments/TIP2018_AUG/models'. </br>
- For example: 'AUG/experiments/TIP2018_AUG/models/DA_wight.pth' </br>

```
python DA_test.py
```


## Examples of Data augmentation

- We can select a $Source Moiré$ image and its corresponding $Source GT$ ground-truth image and a clean image $Target GT$.
- We transfer the Moiré patterns of $Source Moiré$ to $Target GT$ to get $Target Moiré$.
- We can use $Target Moiré$ to do data augmentation.


<img src="./Figure/example_DA.png" width = "400"  div align=center />



## Training for DMSFN

- For perceptual loss , download "[VGG19.pth](https://drive.google.com/file/d/1DcDARBfvK7EczDblnILwqh-NteHn3aBb/view?usp=sharing)"
- Put the VGG19.pth into 'vgg_models'.
- For example: 'vgg_models/VGG19.pth'


Run the following command :
```
python main.py --mode train --data_clear_path datasets/TIP18_crop/TrainData/target --data_moire_path datasets/TIP18_crop/TrainData/source --save_model_path checkpoints/DMSFN_plus
```

## Testing for DMSFN
- For reproducing our results on TIP2018 datasets, download "[TIP_AUG_DMSFN_plus.pth](https://drive.google.com/file/d/1e_BVrk98zxS8Z06B09QoUhAwBPjtMnLC/view?usp=sharing)"
- Put the TIP_DMSFN_plus.pth into 'checkpoints/DMSFN_plus'.
- For example: 'checkpoints/DMSFN_plus/TIP_AUG_DMSFN_plus.pth'

**For testing on TIP18 dataset** </br>

Run the following command :
```
python main.py --mode test --data_clear_path datasets/TIP18_crop/TestData/target --data_moire_path datasets/TIP18_crop/TestData/source --load_model_path checkpoints/DMSFN_plus/TIP_DMSFN_plus --save_results_name DMSFN_plus
```

## Demoireing Results 

<img src="./Figure/table.png" width = "800" div align=center />


<img src="./Figure/vis_results.png" width = "800" div align=center />

## Evaluation
* For evaluation on TIP18 results in MATLAB, download "[TIP_DMSFN_plus_results](https://drive.google.com/drive/folders/1zYDzGhOG617nwmn0cd0yvfo58H62azVx?usp=sharing)" into './results/DMSFN_plus/output'

```
evaluation_TIP18.m
```

## Citation
```

TODO



```
