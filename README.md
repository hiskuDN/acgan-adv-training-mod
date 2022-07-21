# AC-GAN Based Data Agumentation for Adversarial Training

This research proposes data agumentation to adversarial training by using **Unrestricted Adversarial Examples** found in the paper [Constructing Unrestricted Adversarial Examples with Generative Models](https://arxiv.org/abs/1805.07894), NIPS 2018, Montréal, Canada.  

### Training AC-GANs

In order to do unrestricted adversarial attack, we first need a good conditional generative model so that we can search on the manifold of realistic images to find the adversarial ones. You can use `train_acgan.py` to do this. For example, the following command

```bash
CUDA_VISIBLE_DEVICES=0 python train_acgan.py --dataset mnist --checkpoint_dir checkpoints/
```

will train an AC-GAN on the `MNIST` dataset with GPU #0 and output the weight files to the `checkpoints/` directory. 

Run `python train_acgan.py --help` to see more available argument options.

### Unrestricted Adversarial Attack

After the AC-GAN is trained, you can use `main.py` to do targeted / untargeted attack. You can also use `main.py` to evaluate the accuracy and PGD-robustness of a trained neural network classifier. For example, the following command

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --mode targeted_attack --dataset mnist --classifier zico --source 0 --target 1
```

### `For defense by attack mode`
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --mode defense_by_attack --dataset mnist --classifier zico
```

attacks the provable defense method from [Kolter & Wong, 2018](https://arxiv.org/pdf/1711.00851.pdf) on the `MNIST` dataset, with the source class being 0 and target class being 1. 

Run `python main.py --help` to view more argument options. For hyperparameters such as `--noise`, `--lambda1`, `--lambda2`, `--eps`,  `--z_eps`, `--lr`, and `--n_iters` (in that order), please refer to **Table. 4** in the Appendix of our [paper](https://arxiv.org/pdf/1805.07894.pdf). 

### Evaluating Unrestricted Adversarial Examples

In the paper, we use [Amazon Mechanical Turk](https://www.mturk.com/) to evaluate whether our unrestricted adversarial examples are legitimate or not. We have provided `html` files for the labelling interface in folder `amt_websites`.


## Samples

 Perturbation-based adversarial examples (top row) VS unrestricted adversarial examples (bottom-row):

![compare](assets/imgs/compare_adv_imgs.png)

Targeted unrestricted adversarial examples against robust classifiers on `MNIST` (Green borders denote legitimate unrestricted adversarial examples while red borders denote illegimate ones. The tiny white text at the top-left corder of a red image denotes the label given by the annotators. )

![mnist](assets/imgs/mnist_madry_adv_targeted_large_plot.jpg)

