


# DeCNN

This implements training of decoupled model architectures described in DeCNN. Here displays ResNet-34, ResNet-50, and VGG-16.




# ImageNet

This  training is on the ImageNet Classfication Task.


## Requirements

* PyTorch 0.4.0
* cuda && cudnn
* Download the ImageNet dataset and move validation images to labeled subfolders
  * To do this, you can use the following script:
  [https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)


## Training
To train a model, run main.py with the desired model architecture and the path to the ImageNet dataset:

```
python main.py [imagenet-folder with train and val folders] -a resnet34_s3_x15 --lr 0.01
```

The default learning rate schedule starts at 0.01 and decays by a factor of 10 every 30 epochs. 

## Usage
```
usage: main.py [-h] [-a ARCH] [--epochs N] [--start-epoch N] [-b N] [--lr LR]
               [--momentum M] [--weight-decay W] [-j N] [-m] [-p]
               [--print-freq N] [--resume PATH] [-e]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: resnet34 | vgg16_bn |
                        resnet50 | resnet34_s3_x15 | resnet34_s3_x175 |
               resnet50_s3_addchannel | rsnet50_s3_addchannel175 | vgg16_bn_s3_addchannel_v2 
  --epochs N            numer of total epochs to run
  --start-epoch N       manual epoch number (useful to restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        Weight decay (default: 1e-4)
  -j N, --workers N     number of data loading workers (default: 4)
  -m, --pin-memory      use pin memory
  -p, --pretrained      use pre-trained model
  --print-freq N, -f N  print frequency (default: 10)
  --resume PATH         path to latest checkpoitn, (default: None)
  -e, --evaluate        evaluate model on validation set


```


## Result

The results of a single model on ILSVRC-2012 validation set.

<table>
    <tr>
        <th>Model</th>
        <th>top1@prec (val)</th>
        <th>top5@prec (val)</th>
    </tr>
    <tr>
        <th>ResNet-50</th>
        <th>72.72%</th>
        <th>90.90%</th>
    </tr>
    <tr>
        <th>ResNet-50 ×1.5</th>
        <th>72.384%</th>
        <th>90.682%</th>
    </tr>
    <tr>
        <th>ResNet-50 ×1.75</th>
        <th>73.79%</th>
        <th>91.56%</th>
    </tr>
    <tr>
        <th>ResNet-34</th>
        <th>70.26%</th>
        <th>89.678%</th>
    </tr>
    <tr>
        <th>ResNet-34 ×1.5</th>
        <th>69.214%</th>
        <th>88.698%</th>
    </tr>
    <tr>
        <th>ResNet-34 ×1.75</th>
        <th>70.91%</th>
        <th>89.656%</th>
    </tr>
    <tr>
        <th>VGG-16_bn</th>
        <th>73.124%</th>
        <th>91.354%</th>
    </tr>
    <tr>
        <th>VGG-16_bn ×1.75</th>
        <th>72.557%</th>
        <th>90.918%</th>
    </tr>
</table>
