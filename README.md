# Non-autoregressive Transformer for long text generation 

**Note: Uncompleted work in Shanghai ai lab, under guidance of Jiangtao Feng and Fei Yuan**
### Dataset from
iwslt14, wmt raw/distilled: [https://github.com/harvardnlp/cascaded-generatio](https://github.com/harvardnlp/cascaded-generatio)
Europarl/TED/News: [https://github.com/sameenmaruf/selective-attn/tree/master/data](https://github.com/sameenmaruf/selective-attn/tree/master/data), reordered in ./data/prepare-doc

### Preprocess data
    ./run.sh preprocess

### Training
    ./run.sh train

### Evaluation
    ./run.sh eval

