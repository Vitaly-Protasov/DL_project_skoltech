## Skoltech project of the Deep Learning course

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vitaly-Protasov/DL_project_skoltech/)

### Authors of this project (in alphabet order): 
**Alexander Selivanov, Kristina Ivanova, Lucy Airapetyan, Vitaly Protasov.**

---

## What we have done
1) We reimplemented the **original article:** [code2vec by U. Alon et al](https://arxiv.org/pdf/1803.09473.pdf).
2) We improved _F1-score_ on the test dataset of java14m-data [here you can find dataset](https://github.com/tech-srl/code2vec).

|  Our implementation:     |Best F1-score| U. Alon work| 
| ---------------|:-----------:| :-----------:| 
| **Batch size 128 Test**|    0.17671   |  0.1752|  
| **Batch size 128 Validation**| 0.20213| -|  
| **Batch size 1024 Test**| 0.16372     |  -|
| **Batch size 1024 Validation**| 0.1887| -| 
3) __Also, we applied Bert architecture instead of attention layer in the original article.__ Results you can see below:

## If you want to run our code
1) First of all, you can open ipython notebook in colab via the button above. Just run all cells, it's easy to do. 
2) Without notebook in the console:

##### First of all, clone notebook:
```
git clone https://github.com/Vitaly-Protasov/DL_project_skoltech
cd DL_project_skoltech
```
##### In order to download data:
```
./download data.sh
```
##### Start train the NN from the original article:
```
python3 to_train_article_model.py
```
##### Start train improved version with Transformer inside:
```
python3 to_train_bert.py
```

As the parameters which you need to vary are _batch_size_ of validation and train datasets, 
_learning rate_ and _weight decay_ for optimization algorithm.
