# Non-Autoregressive Long Text Generation 

Explore Non-Autoregressive Transformer on Long Text Generation.

**Note: Uncompleted research work in [Shanghai AI Lab](https://www.shlab.org.cn/en), under the guidance of [Jiangtao Feng](https://jiangtaofeng.github.io/) and Fei Yuan**

### Dataset from

Raw/distilled iwslt14/wmt : [https://github.com/harvardnlp/cascaded-generatio](https://github.com/harvardnlp/cascaded-generatio)

Europarl/TED/News: [https://github.com/sameenmaruf/selective-attn/tree/master/data](https://github.com/sameenmaruf/selective-attn/tree/master/data), reordered in ./data/prepare-doc

### Preprocess data
    ./run.sh preprocess

### Training
    ./run.sh train

### Evaluation
    ./run.sh eval

