# MINS
The implement of Multi-Interest News Sequence model.
```bash
Wang, R., Wang, S., Lu, W., Peng, X.: News recommendation via multi-interest
news sequence modelling. In: ICASSP. pp. 7942–7946 (2022)
```

## Requirement

- python~=3.8
- pytorch~=1.9.0
- numpy~=1.20.1
- pandas~=1.2.4
- tensorboard~=2.6
- tqdm~=4.59.0
- nltk~=3.6.2
- scikit-learn~=0.24.1

## Dataset

```bash
# Download GloVe pre-trained word embedding
https://nlp.stanford.edu/data/glove.840B.300d.zip

# Download MIND dataset
https://msnews.github.io/.
```

## Run

```bash
# Train the model, meanwhile save checkpoints
python3 src/train1.py
# Load latest checkpoint and evaluate on the test set
python3 src/evaluate.py
```

## Acknowledgement
Any scientific publications that use our codes and datasets should cite the following paper as the reference:
```bash
@inproceedings{wang2022news,
  title={News recommendation via multi-interest news sequence modelling},
  author={Wang, Rongyao and Wang, Shoujin and Lu, Wenpeng and Peng, Xueping},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7942--7946},
  year={2022},
  organization={IEEE}
}
```

### Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
- Reference https://github.com/yusanshi/NewsRecommendation

