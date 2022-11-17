# ELDA

Official implementation of ELDA: Using Edges to Have an Edge on Semantic Segmentation Based UDA

[Ting-Hsuan Liao](),
[Huang-Ru Liao](),
[Shan-Ya Yang](),
[Jie-En Yao](),
[Li-Yuan Tsao](),
[Hsu-Shen Liu](),
[Bo-Wun Cheng](),
[Chen-Hao Chao](),
[Chia-Che Chang](),
[Yi-Chen Lo](),
[Chun-Yi Lee]()
<br/>
[BMVC 2022](https://bmvc2022.org/programme/papers/) | [arXiv](https://arxiv.org/abs/2211.08888) |

## Installation
```
pip3 install -r requirements.txt
```
## Data Preparation 
Download [Cityscapes](https://www.cityscapes-dataset.com/), [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) and [SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/downloads/).
And change the corresponding data path in ```./data/__init__.py``` .

## Training
Model will be saved under the ```./saved``` directory.
<details>
  <summary>
    <b>1) GTA5 -> Cityscapes</b>
  </summary>

    python3 trainUDA_gta.py --config ./configs/configUDA_gta2city_edge.json --name ELDA_gta

</details>

<details>
  <summary>
    <b>1) SYNTHIA -> Cityscapes</b>
  </summary>
      
    python3 trainUDA_synthia.py --config ./configs/configUDA_syn2city_edge.json --name ELDA_synthia
 
</details>

## Evaluate

<details>
  <summary>
    <b>1) GTA5 -> Cityscapes</b>
  </summary>

    python3 evaluateUDA.py --full-resolution -m deeplabv2_gta --model-path ./checkpoint/gta5_edge_exp/checkpoint.pth
  <p>
    Change [--model-path] to the corresponding checkpoint path.
  </p>

</details>

<details>
  <summary>
    <b>1) SYNTHIA -> Cityscapes</b>
  </summary>
      
    python3 evaluateUDA.py --full-resolution -m deeplabv2_synthia --model-path ./saved/synthia_egde_exp/checkpoint.pth
  <p>
    Change [--model-path] to the corresponding checkpoint path.
  </p>
</details>
Due to the file size limitation, the experiment checkpoint will be released on the official website after publishment.
