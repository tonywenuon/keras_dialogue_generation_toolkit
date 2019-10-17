# keras_dialogue_generation_toolkit

## Introduction

This is a Keras framework for dialogue generation. It includes some basic generative models:

* Seq2Seq with Attention Mechanism ([Neural machine translation by jointly learning to align and translate](https://arxiv.org/pdf/1409.0473.pdf))
* Pointer-Generator ([Get to the point: Summarization with pointer-generator networks](https://arxiv.org/pdf/1704.04368.pdf))
* Memory Network ([End-to-end memory networks](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf))
* Multi-Task ([A knowledge-grounded neural conversation model](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16710/16057))
* Transformer ([Attention is all you need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf))
* Universal Transformer ([Universal transformers](https://arxiv.org/pdf/1807.03819.pdf))
* Transformer with Expanded Decoder ([Enhancing Conversational Dialogue Models with Grounded Knowledge]())

## Structure
1. **commonly_used_code:** 
This folder contains commonly used code for all of the sub-tasks. We can reuse the code in this folder.
2. **configuration:** 
This folder contains configuration of the task, including all kinds of file path. 
3. **run_script:**
The parameter parser file is put in this folder and the run script can be put in this folder. All models' hyper-parameters are set in the args_parser.py.
4. **data:** 
This folder contains all data. For this framework, it has a specific requirement for the data set format. Example data set is in this folder.
5. **pre_precessing:** 
This folder contains data processing code. All process about the data processing should be put into this folder. The final data should be generated to the data folder.
6. **example:** 
For all of the models listed above, the training entrance is in this folder. You can get start from this folder.
7. **models:**
The core model files are in this folder. You can get the code of each model.

## Reference and Acknowledgement:
1. Pointer_generator: Thank Abisee, we use his repo on the Github. Obeying the Apache License version 2.0, we just use this repo for researching. I refer readers to the original repo: [Pointer-Generator](https://github.com/abisee/pointer-generator)
2. Memory Neural Network: Thank these authors. We inspire from two repos, [memn2n tensorflow version](https://github.com/domluna/memn2n) and [memn2n keras version](https://github.com/IliaGavrilov/ChatBotEndToEndMemoryNeuralNet)
3. Transformer and Universal Transformer: Thanks kpot for his repo: keras-transformer. I changed his original code to fit with my experiments. He also implemented BERT on top of this repo. Even this repo didn't implement entire Transformer, it still easy to add Decoder part in this framework. The original repo can be found here: [keras-transformer](https://github.com/kpot/keras-transformer).

## Usage:
Go into the example folder and run the 'train_\*\*\*.py' file. An example is given below:

```
python train_transformer.py --data_set=wizard \
       --exp_name=transformer \
       --batch_size=40 \
       --src_seq_length=30 \
       --tar_seq_length=30 \
       --early_stop_patience=2 \
       --lr_decay_patience=1 \
       --lr=0.001
```
As for the hyper-parameters, they can be found in `run_script\args_parser.py`.
