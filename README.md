# Universal Adversarial Training Based on ACGAN-based Adversarial Attack Generation

This research proposes data agumentation to Adversarial Training by using Adversarial Examples generated using AC-GAN generated **Unrestricted Adversarial Examples**.

### Training Unrestrcited Adversarial Examples using AC-GANs
AC-GAN archtiecture is based on the one proposed in the paper [Constructing Unrestricted Adversarial Examples with Generative Models](https://arxiv.org/abs/1805.07894), NIPS 2018, Montréal, Canada. The commands

```bash
CUDA_VISIBLE_DEVICES=0 python train_acgan.py --dataset mnist --checkpoint_dir checkpoints/
CUDA_VISIBLE_DEVICES=0 python train_acgan.py --dataset celebA --checkpoint_dir checkpoints-celebA/
```

will train an AC-GAN on the `MNIST` or `CelebA` dataset and output the weight files to the `checkpoints/` directory. 

Run `python train_acgan.py --help` to see more available argument options.

### Unrestricted Adversarial Attack for Universal Adversarial Training

### `Modified from the default generation mode`
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --mode defense_by_attack --dataset mnist --classifier zico --test_num 0
CUDA_VISIBLE_DEVICES=0 python main.py --mode defense_by_attack --dataset celebA --test_num 0
```

* run the following before running defense by attack if you're using a docker container \
```bash
pip install opencv-python
apt-get update -y
apt-get install -y libgl1-mesa-glx
pip install tf_slim
pip install imageio
```

Run `python main.py --help` to view more argument options. For hyperparameters such as `--noise`, `--lambda1`, `--lambda2`, `--eps`,  `--z_eps`, `--lr`, `--n_iters`, and `--test_num` (in that order)


Results can be found on the published paper (link coming soon!)
