<!-- # ReasonPlan: Unified Scene Prediction and Decision
Reasoning for Closed-loop Autonomous Driving -->

<div align="center">
<h3>[🎉CoRL 2025] ReasonPlan: Unified Scene Prediction and Decision Reasoning <br>for Closed-loop Autonomous Driving</h3>

Xueyi Liu<sup>1,2,3</sup>, Zuodong Zhong<sup>4</sup>, Junli Wang<sup>1,2</sup>, Yuxin Guo<sup>1,2</sup>, Zhiguo Su<sup>3</sup>, Qichao Zhang<sup>1,2\*</sup>, <br> Yun-Fu Liu<sup>3</sup>, Yinfeng Gao<sup>4</sup>, Yupeng Zheng<sup>1,2</sup>,  Qiao Lin<sup>3</sup>, Huiyong Chen<sup>3</sup>, Dongbin Zhao<sup>1,2\*</sup>

<sup>1</sup>  SKL-MAIS, Institute of Automation, Chinese Academy of Sciences, <br><sup>2</sup>  School of Artificial Intelligence, University of Chinese Academy of Sciences, <br><sup>3</sup>  EACON, <br><sup>4</sup>  School of Automation and Electrical Engineering, University of Science and Technology Beijing


(\*) Corresponding author.

<a href="https://arxiv.org/pdf/2505.20024"><img src='https://img.shields.io/badge/arXiv-ReasonPlan-red' alt='Paper PDF'></a>
<a href="https://github.com/Liuxueyi/ReasonPlan/"><img src='https://img.shields.io/badge/Project_Page-ReasonPlan-green' alt='Project Page'></a>
<a href="https://huggingface.co/datasets/LiuxyIA/ReasonPlan_PDR/tree/main/"><img src='https://img.shields.io/badge/Dataset-PDR-blue' alt='Dataset'></a>
</div> 

## Abstract
Due to the powerful vision-language reasoning and generalization abilities, multimodal large language models (MLLMs) have garnered significant attention in the field of end-to-end (E2E) autonomous driving. However, their application to closed-loop systems remains underexplored, and current MLLM-based methods have not shown clear superiority to mainstream E2E imitation learning approaches. In this work, we propose ReasonPlan, a novel MLLM fine-tuning framework designed for closed-loop driving through holistic reasoning with a self-supervised Next Scene Prediction task and supervised Decision Chain-of-Thought process. This dual mechanism encourages the model to align visual representations with actionable driving context, while promoting interpretable and causally grounded decision making. We curate a planning-oriented decision reasoning dataset, namely PDR, comprising 210k diverse and high-quality samples. Our method outperforms the mainstream E2E imitation learning method by a large margin of 19% L2 and 16.1 driving score on Bench2Drive benchmark. Furthermore, ReasonPlan demonstrates strong zero-shot generalization on unseen DOS benchmark, highlighting its adaptability in handling zero-shot corner cases. 

## Table of Contents:
1. [Conda Environment Setup](#conda-environment-setup)
2. [Dataset and Pretrained Model Configuration](#dataset-and-pretrained-model-configuration)
3. [Training](#training)
4. [Evaluation](#Open-Loop-Evaluation)

## Conda Environment Setup:
### Create Conda Environment
```bash
git clone https://githubfast.com/Liuxueyi/ReasonPlan.git
conda create -n reasonplan python=3.8 -y
conda activate reasonplan
pip install --upgrade pip
```
### Install PyTorch and CUDA
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
### Install MMCV
```bash
git clone https://githubfast.com/Thinklab-SJTU/Bench2DriveZoo.git
cd Bench2DriveZoo
pip install ninja packaging
pip install -v -e .
```
### Install LLaVA Environment
```bash
cd ..
pip install -e ".[train]"
pip install pillow==10.2.0

wget https://githubfast.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

pip install accelerate==0.28.0
```
## Dataset and Pretrained Model Configuration
### Download Datasets and Models
```bash
pip install -U huggingface_hub
huggingface-cli download --repo-type dataset --resume-download rethinklab/Bench2Drive --local-dir Bench2Drive
huggingface-cli download --repo-type model --resume-download google/siglip-so400m-patch14-384 --local-dir siglip
huggingface-cli download --repo-type model --resume-download lmms-lab/llava-next-interleave-qwen-0.5b --local-dir llava-qwen0.5B
```

### Create Symbolic Links
```bash
mkdir data
ln -s /path/to/bench2drive ./data
ln -s /path/to/siglip ./google
ln -s /path/to/qwen ./llava-next-interleave-qwen-0.5b
```

## Training
Please modify the corresponding parameters or configuration.
```bash
bash scripts/train_stage1.sh
bash scripts/train_stage2.sh
```
### Note
Modify the following file in your environment 
`/path_to_your_conda/envs/reasonplan/lib/python3.8/site-packages/transformers/models/qwen2/modeling_qwen2.py`

Change **line 1210** from
```python
hidden_states = outputs.hidden_states
```
to
```python
hidden_states = hidden_states
```

## Open Loop Evaluation
Please modify the corresponding parameters or configuration.
```bash
bash scripts/eval_multi.sh
```

## Cite
```
@inproceedings{liu2025reasonplan,
      title={ReasonPlan: Unified Scene Prediction and Decision Reasoning for Closed-loop Autonomous Driving}, 
      author={Xueyi Liu and Zuodong Zhong and Junli Wang and Yuxin Guo and Zhiguo Su and Qichao Zhang and Yun-Fu Liu and Yinfeng Gao and Yupeng Zheng and Qiao Lin and Huiyong Chen and Dongbin Zhao},
      booktitle={9th Conference on Robot Learning},
      year={2025},
}
```