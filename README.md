#Neural Sentiment Translator

## Installation

Require python 2.7, Tensorflow, Gensim (for word2vec model), and Numpy

dowload dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract into data/

Download google news negative 300 vectors and extract into model/

## Usage

### Training

```  
python translate.py
```

To train with GPU, add the --gpu flag:

```  
python translate.py --gpu
```

To load a checkpoint, use --load (and --skip_train to skip the training step)

```  
python translate.py --skip_train --load path/to/checkpoint/dir
```

To get the sentiment of a sentence, use the --custom_input flag:

```  
python translate.py --skip_train --load path/to/checkpoint/dir --custom_input "input sentence"
```

To translate to a given sentiment use --target_sentiment (-1 or 0 is negative, 1 is positive):

```  
python translate.py --skip_train --load path/to/checkpoint/dir --custom_input "input sentence" --target_sentiment [-1,1]
```

