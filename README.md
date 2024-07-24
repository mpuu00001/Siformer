<div align="center">
<h1>
<b>
Siformer: Feature Isolated Transformer for Efficient Skeleton-based Sign Language Recognition
</b>
</h1>
</div>

## Installation
```shell
conda create -n siformer python==3.11
conda activate siformer

# Please install PyTorch according to your CUDA version.
pip install -r requirements.txt
```

## Get Started
![method](https://github.com/user-attachments/assets/d3632363-7f39-448d-992c-92a75dc70eef)

To train the model, just specify the hyperparameters and execute the following command:

```
python -m train
  --experiment_name [str; name of the experiment to name the output logs and plots]
  --epochs [int; number of epochs]
  --lr [float; learning rate]
  --num_classes [int; the number of classes to be recognised by the model]
  
  --attn_type [str; the attention mechanism used by the model]
  --num_enc_layers [int; determines the number of encoder layer]
  --num_dec_layers [int; determines the number of decoder layer]
  --FIM [boolean; determines whether feature-isolated mechanism will be applied]
  --PBEE_encoder [bool; determines whether patience-based encoder will be used for input-adaptive inference]
  --PBEE_decoder [bool; determines whether patience-based decoder will be used for input-adaptive inference]
  --patience [int; determines the patience for earlier exist]
  
  --training_set_path [str; the path to the CSV file with training set's skeletal data]
  --validation_set_path [str; the Path to the CSV file with validation set's skeletal data]
  --testing_set_path [str; the path to the CVS file with testing set's skeletal data]
```

If you leave the paths for either the validation or testing sets empty, the corresponding metrics will not be computed. 
Additionally, we offer a pre-defined parameter to automatically divide the validation set according to a desired split 
of the training set. You can find detailed descriptions of these and various other specific hyperparameters in the 
[train.py](https://github.com/mpuu00001/Skeleton_based_SLR/blob/main/train.py) file. 
Each of these hyperparameters has a default value that we have determined to work well in our experiments.

Here are some examples of usage:
```
python -m train --experiment_name LSA64 --training_set_path datasets/LSA64_60fps.csv --num_classes 64 --experimental_train_split 0.8 --validation_set split-from-train --validation_set_size 0.2 
```

```
python -m train --experiment_name WLASL100 --training_set_path datasets/WLASL100_train_25fps.csv --validation_set_path datasets/WLASL100_val_25fps.csv --validation_set from-file --num_classes 100
```

## Dataset
We employed the [WLASL100 and LSA64 datasets](https://drive.google.com/drive/folders/13JyaGqX4voqC1wv3ETzjdE_wh50uLszv?usp=sharing) for our experiments. Their corresponding citations can be found below:
```
@inproceedings{li2020word,
    title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
    author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
    booktitle={The IEEE Winter Conference on Applications of Computer Vision},
    pages={1459--1469},
    year={2020}
}
```
```
@inproceedings{ronchetti2016lsa64,
    title={LSA64: an Argentinian sign language dataset},
    author={Ronchetti, Franco and Quiroga, Facundo and Estrebou, C{\'e}sar Armando and Lanzarini, Laura Cristina and Rosete, Alejandro},
    booktitle={XXII Congreso Argentino de Ciencias de la Computaci{\'o}n (CACIC 2016).},
    year={2016}
}
```

## License
Soon...  

## Citation
This work has been accepted by ACM MM' 24. We will add the full citation upon publication.

