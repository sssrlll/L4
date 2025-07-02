# L4: Mutual Learning Helps Lifelong Language Learning

Code for the paper "[L4: Mutual Learning Helps Lifelong Language Learning]"  

Our code is based on the released code from [Lifelong Language Knowledge Distillation](https://github.com/voidism/L2KD). Most of the settings are identical to theirs.

## Dataset

| Task | Dataset (Original Data Link) |
| ---- | ------- |
| Summarization | [CNN/DM](https://cs.nyu.edu/~kcho/DMQA/) |
| Goal-Oriented Dialogue | [WOZ](https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz) |
| Semantic Parsing | [WikiSQL](https://github.com/salesforce/WikiSQL) |
| Natural Language Generation | [E2ENLG](https://github.com/tuetschek/e2e-dataset) |
| Natural Language Generation | [RNNLG](https://github.com/shawnwun/RNNLG) |
| Text Classification | [AGNews, Yelp, Amazon, DBPedia, Yahoo](http://goo.gl/JyCnZq) |

We use the released data from L2KD's authors [here](https://www.dropbox.com/s/t51qq9lzz0gtg7m/l2kd_data.zip).

## Dependencies (same as L2KD)
python packages are listed in `requirements.txt`

## 🔧 Setup (same as L2KD)
1. Create the following two directories in wherever you want. (you can name the directories arbitrarily):
    - `data directory`: Where the dataset will be load by the model.
    - `model directory`: The place for the model to dump its outputs.
2. Download the dataset into `data directory`.
3. Make a copy of `env.example` and save it as `env`. In `env`, set the value of DATA_DIR as `data directory` and set the value of  MODEL_ROOT_DIR as `model directory`.

### Examples

See examples in `mutual_WCS.sh`.
