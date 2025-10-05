# MC-CrackDA

Code for the paper **MC-CrackDA: A Monte Carlo Dropout-based Domain Adaptation Method for Crack Segmentation**.

---

## Pretrained Models

* **Source domain trained model:**
  [Download Link](https://pan.baidu.com/s/1yODAX0j8h-GSOtgIoaxZrg?pwd=ascv)

* **Domain adaptation: CrackSeg9K → Roboflow-Crack:**
  [Download Link](https://pan.baidu.com/s/1EgWx_nNUzNEjaLg5lQP54g?pwd=22kx)

* **Domain adaptation: CrackSeg9K → UAV-Crack:**
  [Download Link](https://pan.baidu.com/s/1kfBEQ17cFAZ9PDcQCWJU6Q?pwd=dkpc)

---

## Environment Setup

First, create and activate a conda virtual environment:

```bash
conda create -n MC-CrackDA python==3.9 -y
conda activate MC-CrackDA
```

Then, install PyTorch, Torchvision, and other dependencies:

```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install albumentations timm
```

---

## Notes for Users in China

If you encounter issues downloading from the default PyTorch repository, you may use the mirror:

```bash
pip install torch==2.1.2+cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu111/?mirror_intel_list
pip install torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install albumentations timm
```

---

