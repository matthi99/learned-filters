# Learned filters


# Introduction

Let $`\mathbf{A}: \mathbb{X} \rightarrow \mathbb{Y}`$  be a bounded linear operator between two Hilbert spaces $`\mathbb{X}`$ and $`\mathbb{Y}`$. We consider the inverse problem
```math
\mathbf{A} x^+ + z =y^{\delta}
```
where $`z`$ is the data pertubation with $` \|z\| \leq \delta`$ for some noise level  $`\delta >0, y^{\delta}`$ is the given noisy data, and we aim to find a stable solution for the signal $`x^+`$. Inverting the operator $`\mathbf{A}`$ is ill-posed in the sense that the Moore-Penrose inverse  $`\mathbf{A}^+`$ is discontinuous, so small errors in the data could significantl enlarge during the solution procedure. To adress this issue regularization methods have bee developed with the goal of finding an approximate but stable solution. 

If the operator  $`\mathbf{A}`$ has a diagonal frame decomposition (DFD) $`(u_\lambda, v_\lambda, \kappa_\lambda)_{\lambda \in \Lambda}`$ a strategy for regularizing inverse problems is the use of non-linear filtered diagonal frame decomposition
```math
\mathcal{F}_\alpha(y^\delta) := \sum_{\lambda \in \Lambda} \frac{1}{\kappa_\lambda} \varphi_\alpha(\kappa_\lambda, \langle y^\delta,  v_\lambda \rangle) \bar{u}_\lambda
```
where $`\varphi_\alpha: \mathbb{R}_+ \times \mathbb{R} \rightarrow \mathbb{R}`$ is a non-linear regularizing filter. (Cite paper) shows that if the filters $`\varphi_\alpha`$ meet certain conditions the non-linear DFD is a convergent regularization method. 

Although some of the required properties for the filters $`\varphi`$ are intuitively quite reasonable, it is not clear at all how an optimal non linear filter should look like. This repoitory is trying to adress this issue by learning non-linear regularizing filters using neural networks for specific examples. To be precize the Radon transform is chosen as the forward operator and the noise $`z`$ is assumed to be white noise. The available DFDs of the radon transform are all based on wavelet transforms. 




# Instalation

1. Download the [SARS-COV-2 Ct-Scan Dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset). Make shure the data set is saved in this structure
``` 
DATASET_FOLDER/
├── COVID 
├── non-COVID
```
Note that the COVID folder is optional and does not have to be present.

2. Clone the git repository. 
```
git clone https://git.uibk.ac.at/c7021123/learned-filters.git
``` 

3. Intall and activate the virtual environment.
```
cd learned-filters
conda env create -f env_filter.yml
conda activate filter
``` 

# Usage

## Preprocessing

To prepare the downloaded dataset for training run the following command in your console
```
python3 create_data.py DATASET_FOLDER --number XXX
``` 
With `--number` you can specify how many pairs od training data should be generated (`default=500`). The maximal number which can be chosen is 1229. 
When running `create_data.py` the preprocessed images will be saved in your `DATASET_FOLDER`. 

## Training

To train the framework run the command
```
python3 train.py --wave 'WAVELET' --levels XXX --s2n-ratio XXX --N_epochs XXX
``` 
- `--wave` specifies based on which wavelet the DFD of the Radon tranform should be performed (`default='wave'`). The code uses the Pytorch Wavelet Toolbox (ptwt) which supports discrete wavelets, see also `pywt.wavelist(kind='discrete')`. 
- `--levels` specifies up to how many levels the wavelet decomposition should be performed (`default=8`). Since the size of the preprocessed images in the training set is $`256 \times 256`$ it has to be an integer between 1 and 8. 
- `--s2n_ratio` choses what what signal-to-noise ratio the filters should be learned (`default=8`). Possible signal-to-noise ratios are 2,4,8,16,32,64,128,256, 512 
- `--N_epochs` defines for how many epochs the networks should be trained (`default=100`)

## Testing

To test the final regularization model on your own images run
```
python3 test.py INPUT_FOLDER OUTPUT_FOLDER --wave 'WAVELET' --levels XXX --s2n-ratio XXX 

``` 
The images in the `INPUT_FOLDER` should be of PNG or JPG format and of course the chosen configuation must have been trained beforehand. Also select a separate `OUTPUT_FOLDER` for each configuration!


## Authors and acknowledgment
Matthias Schwab<sup>1,2</sup>, Andrea Ebner<sup>1</sup>

<sup>1</sup> Department of Mathematics, University of Innsbruck, Technikerstrasse 13, 6020 Innsbruck, Austria

<sup>2</sup> University Hospital for Radiology, Medical University Innsbruck, Anichstraße 35, 6020 Innsbruck, Austria



