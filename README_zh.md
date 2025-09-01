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

## 目录:
1. [配置conda环境](#配置conda环境)
2. [数据集配置](#数据集及预训练模型配置)
3. [训练](#训练)
4. [评估](#开环评估)

## 配置conda环境：
### 创建conda环境
```bash
git clone https://githubfast.com/Liuxueyi/ReasonPlan.git
conda create -n reasonplan python=3.8 -y
conda activate reasonplan
pip install --upgrade pip
```
### 安装torch和cuda
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
### 安装mmcv
```bash
git clone https://githubfast.com/Thinklab-SJTU/Bench2DriveZoo.git
cd Bench2DriveZoo
pip install ninja packaging
pip install -v -e .
```
### 安装llava环境
```bash
cd ..
pip install -e ".[train]"
pip install pillow==10.2.0

wget https://githubfast.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

pip install accelerate==0.28.0
```
## 数据集及预训练模型配置
### 数据集及模型下载
```bash
pip install -U huggingface_hub
huggingface-cli download --repo-type dataset --resume-download rethinklab/Bench2Drive --local-dir Bench2Drive
huggingface-cli download --repo-type model --resume-download google/siglip-so400m-patch14-384 --local-dir siglip
huggingface-cli download --repo-type model --resume-download lmms-lab/llava-next-interleave-qwen-0.5b --local-dir llava-qwen0.5B
```

### 创建软链接
```bash
mkdir data
ln -s /path/to/bench2drive ./data
ln -s /path/to/siglip ./google
ln -s /path/to/qwen ./llava-next-interleave-qwen-0.5b
```

## 训练
请修改相应参数或配置
```bash
bash scripts/train_stage1.sh
bash scripts/train_stage2.sh
```
### Note
修改文件
`/path_to_your_conda/envs/reasonplan/lib/python3.8/site-packages/transformers/models/qwen2/modeling_qwen2.py`

将第1210行从
```python
hidden_states = outputs.hidden_states
```
改为
```python
hidden_states = hidden_states
```

## 开环评估
请修改相应参数或配置
```bash
bash scripts/eval_multi.sh
```

