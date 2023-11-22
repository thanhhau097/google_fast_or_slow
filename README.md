# Google Fast or Slow
This repository contains the code for the first place solution on Google - Fast or Slow? Predict AI Model Runtime competition hosted in Kaggle.
For more information about the competition please access the [link](https://www.kaggle.com/competitions/predict-ai-model-runtime)

The solution overview can be accessed [here](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456343)

## Initial setup
Start by cloning this repo to your directory,
```
git clone https://github.com/thanhhau097/google_fast_or_slow
cd google_fast_or_slow
```
Then, assuming you have Anaconda installed in your computer, create a new environment and install the requirements.
```
conda create -n kaggle_tpu python=3.10
conda activate kaggle_tpu
pip install -r  requirements.txt
```

## Data preprocessing
We used the dataset below for training in the layout datasets, which is identical to the original dataset except changing the padding to `-1` in [this line](https://github.com/google-research-datasets/tpu_graphs/blob/main/tpu_graphs/process_data/xla/featurizers.h#L137).

Instead of re-generating all data you can directly download the dataset [here](https://www.kaggle.com/datasets/tomirol/layout-npz-padding). Please note that this is important, running our pre-trained models with the original competition data will yield bad results.

Prepare the data (`-1` padding, graph prunning and compression) by running the following instructions:
```
# assuming kaggle credentials are set, download the data
cd data
kaggle competitions download -c predict-ai-model-runtime
kaggle datasets download tomirol/layout-npz-padding

unzip predict-ai-model-runtime.zip
unzip layout-npz-padding.zip
rm *.zip

rm -rf ./npz_all/npz/layout
mv ./layout_npz ./npz_all/npz/layout

python data_compression.py
cd ../
```

Download and unzip the model weights to reproduce the solution

```
kaggle datasets download tomirol/tile-models-google-runtime
kaggle datasets download arc144/google-fast-slow-viet-br-connection-weights

unzip tile-models-google-runtime.zip
unzip google-fast-slow-viet-br-connection-weights.zip
rm tile-models-google-runtime.zip
rm google-fast-slow-viet-br-connection-weights.zip
```

## Inference
In order to predict with our models and generate the `submission.csv` file you can run `./predict.sh` after having followed the steps above.
The final output will be a filled titled `submission.csv` located at the root of this repository.


## Training:
For each type, we train 5-20 models for each collection with different seeds and folds, then ensemble the predictions result by `mean` aggregation.
To train each collection please run the corresponding script:

- Tile XLA:
`./train_tile.sh`

- Layout:XLA:Default
`./train_xla_default.sh`

- Layout:XLA:Random
`./train_xla_random.sh`

- Layout:NLP:Default
`./train_nlp_default.sh`

- Layout:XLA:Random
`./train_nlp_random.sh`

Note that after training the ensemble we manually remove any folds that might have had a very bad seed, i.e., too low validation accuracy when compared to others. 


### Sources
1. Tile XLA weights: https://www.kaggle.com/datasets/tomirol/tile-models-google-runtime
2. Layout weights: https://www.kaggle.com/datasets/arc144/google-fast-slow-viet-br-connection-weights/data
3. Layout dataset (`-1` padded): https://www.kaggle.com/datasets/tomirol/layout-npz-padding