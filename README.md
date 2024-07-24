<div align="center">
<h1>
<b>
Siformer: Feature Isolated Transformer for Efficient Skeleton-based Sign Language Recognition
</b>
</h1>
</div>

## Get Started

First, ensure that you install all required dependencies by following command:

```shell
pip install -r requirements.txt
```

To train the model, just specify the hyperparameters and execute the following command:

```
python -m train
  --experiment_name [str; name of the experiment to name the output logs and plots]
  --epochs [int; number of epochs]
  --lr [float; learning rate]
  
  --attn_type [str; the attention mechanism used by the model]
  --num_enc_layers [int; Determines the number of encoder layer]
  --num_dec_layers [int; Determines the number of decoder layer]
  --FITR [boolean; Determines the patience for earlier exist]
  --PBEE_encoder [bool; Determines whether patience-based encoder will be used for input-adaptive inference]
  --PBEE_decoder [bool; Determines whether patience-based decoder will be used for input-adaptive inference]
  --patience [int; Determines the patience for earlier exist]
  
  --training_set_path [str; path to the csv file with training set's skeletal data]
  --validation_set_path [str; path to the csv file with validation set's skeletal data]
  --testing_set_path [str; path to the csv file with testing set's skeletal data]
```

If you leave the paths for either the validation or testing sets empty, the corresponding metrics will not be computed. 
Additionally, we offer a pre-defined parameter to automatically divide the validation set according to a desired split 
of the training set. You can find detailed descriptions of these and various other specific hyperparameters in the 
[train.py](https://github.com/nebularer/Siformer/blob/main/train.py) file. 
Each of these hyperparameters has a default value that we have determined to work well in our experiments.

## Architecture
Soon...  

## Data
We will make the datasets publicly available upon publication.

## License
Soon...  

## Citation
This work has been accepted by ACM MM' 24. We will add the full citation upon publication.

