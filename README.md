# Learned filters

This repository is created for learning non-linear regularizing filters for inverting the Radon transform. If you use this repository please cite:

```
Ebner, A., Schwab, M., & Haltmeier, M. (2025). Error Estimates for Weakly Convex Frame-Based Regularization Including Learned Filters. SIAM Journal on Imaging Sciences, 18(2), 822-850.
```

# Introduction

Let $`\mathbf{A}: \mathbb{X} \rightarrow \mathbb{Y}`$  be a bounded linear operator between two Hilbert spaces $`\mathbb{X}`$ and $`\mathbb{Y}`$. We consider the inverse problem
```math
\mathbf{A} x^+ + z =y^{\delta},
```
where $`z`$ is the data pertubation with $` \|z\| \leq \delta`$ for some noise level  $`\delta >0, y^{\delta}`$ is the given noisy data, and we aim to find a stable solution for the signal $`x^+`$. Inverting the operator $`\mathbf{A}`$ is ill-posed in the sense that the Moore-Penrose inverse  $`\mathbf{A}^+`$ is discontinuous, so small errors in the data could significantly enlarge during the solution procedure. To adress this issue regularization methods have bee developed with the goal of finding an approximate but stable solution. 

If the operator  $`\mathbf{A}`$ has a diagonal frame decomposition (DFD) $`(u_\lambda, v_\lambda, \kappa_\lambda)_{\lambda \in \Lambda}`$ a strategy for regularizing inverse problems is the use of non-linear filtered DFD
```math
\mathcal{F}_\alpha(y^\delta) := \sum_{\lambda \in \Lambda} \frac{1}{\kappa_\lambda} \varphi_\alpha(\kappa_\lambda, \langle y^\delta,  v_\lambda \rangle) \bar{u}_\lambda
```
where $`\varphi_\alpha: \mathbb{R}_+ \times \mathbb{R} \rightarrow \mathbb{R}`$ is a non-linear regularizing filter. One can show that if the filters $`\varphi_\alpha`$ meet certain conditions the non-linear DFD is a convergent regularization method. 

Although some of the required properties for the filters $`\varphi_\alpha`$ are intuitively quite reasonable, it is not clear at all how an optimal non-linear filter should look like. This repoitory is trying to adress this issue by learning non-linear regularizing filters using neural networks for specific examples. To be precise the Radon transform is chosen as the forward operator $`\mathbf{A}`$ and the noise $`z`$ is assumed to be Gaussian white noise. The available DFDs of the Radon transform are all based on wavelet transforms and the method is trained and tested on CT scans. 


# Instalation

1. Clone the git repository. 
```
git clone https://git.uibk.ac.at/c7021123/learned-filters.git
``` 

2. Intall and activate the virtual environment.
```
cd learned-filters
conda create -n filter python=3.9
conda activate filter
pip install -r requirements.txt
``` 

# Usage

## Preprocessing
1. Download the [SARS-COV-2 Ct-Scan Dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset). Make shure the data set is saved in this structure
``` 
DATASET_FOLDER/
├── COVID 
├── non-COVID
```
Note that the COVID folder is optional and does not have to be present.

2. Prepare the downloaded dataset for training. For this run the following command in your console
```
python create_data.py --path DATASET_FOLDER --noise NOISE --number XXX
``` 
With `--number` you can specify how many pairs of training data should be generated (`default=500`). The maximal number which can be chosen is 1229.  
When running `create_data.py` the preprocessed images will be saved in your `DATASET_FOLDER`. 

## Training

To train the framework run the command
```
python train.py --wave "WAVELET" --levels XXX --alpha XXX --N_epochs XXX --type "TYPE"
``` 
- `--wave` specifies based on which wavelet the DFD of the Radon tranform should be performed (`default='haar'`). The code uses the Pytorch Wavelet Toolbox (ptwt) which supports discrete wavelets, see also `pywt.wavelist(kind='discrete')`. 
- `--levels` specifies up to how many levels the wavelet decomposition should be performed (`default=8`). Since the size of the preprocessed images in the training set is $`256 \times 256`$ it has to be an integer between 1 and 8. 
- `--alpha` specifies the noise level in whcih should be trained (`default=4`). Possible noise levels are 32,28,24,20,16,12,8,4,0. 
- `--N_epochs` defines for how many epochs the networks should be trained (`default=20`)
- `--type` defines which contraints should be applied to the filters. You can choose between porposed, unconstrained, nonexpansice or linear. (`default=proposed`)

## Testing

To test the final regularization model on your own images run
```
python test.py --wave "WAVELET" --levels XXX --alpha XXX --types "TYPES"

```
The test images should be saved in the DATASET_FOLDER in a subfolder called "testset". The images should be of PNG or JPG format and of course the chosen configuation must have been trained beforehand. Results are saved in the corresponding folders were the training process was saved. 


## Authors and acknowledgment
Matthias Schwab<sup>1,2</sup>, Andrea Ebner<sup>1</sup>

<sup>1</sup> Department of Mathematics, University of Innsbruck, Technikerstrasse 13, 6020 Innsbruck, Austria

<sup>2</sup> University Hospital for Radiology, Medical University Innsbruck, Anichstraße 35, 6020 Innsbruck, Austria



