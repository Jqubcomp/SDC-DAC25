# On-Device Multi-modal Generative AI for Science Contest at DAC 2025

## Introduction

The On-Device Multi-modal Generative AI for Science Contest at [DAC'25](https://www.dac.com/) is a challenging, 
multi-month research and development competition focused on real-world scientific problems that require deploying multi-modal generative AI algorithms on edge devices. 
It is open to teams from around the world. Winning teams will be invited to attend [DAC'25](https://www.dac.com/) to receive their awards.

This is a demo repo for a base implementation of multi-modal generative model.
This implementation is based on the "[A Novel Diffusion Model for Pairwise Geoscience Data Generation with Unbalanced Training Dataset](https://arxiv.org/abs/2501.00941)",
and code is built upon the InversionNet (https://github.com/lanl/OpenFWI), and Diffusion (https://github.com/lucidrains/denoising-diffusion-pytorch, https://github.com/Hramchenko/diffusion_distiller). We gratefully acknowledge these open-source contributions.




## Dependencies

### Prerequisites
- Python 3.9.9
- PyTorch 2.6.0

Install the necessary packages based on the prerequisites.
```bash
pip install -r requirements.txt
```

## Datasets
The dataset can be downloaded from openFWI, and the [CurveFault-A](https://drive.google.com/drive/folders/1vqUHJ-iRwp3ozL-e4HhKGpdO0e7NQZE1) will be used for this contest.

Each data file contains 500 seismic wave samples or velocity maps.

The **seismic data file** has a shape of `(500, 5, 1000, 70)`, where:  
- `500` indicates the number of seismic wave samples,  
- `5` represents the number of sources (can be considered as channels),  
- `1000` is the time dimension,  
- `70` corresponds to the offset.

The **velocity map file** has a shape of `(500, 1, 70, 70)`, where:  
- `500` refers to the number of velocity maps,  
- `1` is the channel dimension,  
- the two `70`s represent depth and offset, respectively.

To learn more about the dataset and how to visualize it, please refer to this [tutorial](https://colab.research.google.com/drive/17s5JmVs9ABl8MpmFlhWMSslj9_d5Atfx?usp=sharing).


## The Framework
![Framework](./imgs/overview.png)
We follow the framework designed in "[A Novel Diffusion Model for Pairwise Geoscience Data Generation with Unbalanced Training Dataset](https://arxiv.org/abs/2501.00941)" (shown above) to build the framework.


## Training
### Encoder-Decoder Training
Our code of training encoder and decoder is built upon the [InversionNet](https://github.com/lanl/OpenFWI), https://github.com/lanl/OpenFWI and ViT (Vision Transformer).


Train encoder and decoder using veloctity map and paired corresponding seismic wave. It will use the self-supervised 
learning and supervised learning to build a latent space and encoder and two decoders. 
Run:
```bash
cd encoder_decoder 
python train_EncDec.py
```
Use the '--train-data' to set the path of seismic data, 'train-label' to set the path of velocity map,
'--num_data' to set the training number size, '--epoch_block' and '--val_every' to set the epochs and frequency to save checkpoint.

For example, you can run:
```bash
python train_EncDec.py --num_data 48000 --latent_dim 128
```
with 48,000 paired data with 128 dimension of latent space.

### Diffusion training
Our code for diffusion training is built upon https://github.com/lucidrains/denoising-diffusion-pytorch
and https://github.com/Hramchenko/diffusion_distiller

Run: 
```bash
cd ../diffusion
python train_diff.py
```
Use the '--num_data' set the training number size, '--train_data' to set the path of seismic data, 'train_label' to set the path of velocity map.

For example, you can run:
```bash
python train_diff.py --num_data 48000 --latent_dim 128
```
with 48,000 data with 128 dimension of input variable to train diffusion model.


### Generation
After diffusion model is trained, run the following to generate velocity map and seismic waveform simultaneously. 
Run: 
```bash
python generation.py
```
the generated data will be store at 'gen_data'.

Keep a similar setting as train_diff.py.
Use the '--model_file' to change the checkpoint wish to load. '--num_samples' is set default as 10000 (with 500 per file with same size as the original dataset). (Note that, load the dataset just for convenience when generation.)

For example, you can run:
```bash
python generation.py --model_file 5 --num_samples 10000
```
to generate 10000 paired seismic wave samples and velocity maps with saved model-5. 

## Important Note
Due to time constraints, we encourage all teams to focus on optimizing their machine learning algorithms.  
Therefore, **only Python-based solutions will be accepted**.

All package versions must strictly follow the specifications in the **Prerequisites** section and the provided `requirements.txt` file. Additional python-based packages can be used but please make sure the version of all packages can be accomodated on both HPC (for performance evaluation purpose) and edge device (for latency evaluation purpose).

The Raspberry Pi 4 running Ubuntu 23.04 (64-bit) with an aarch64 architecture will be used for latency evaluation. 
Please ensure that your solution is compatible with this environment.


We will run the provided `generation.py` script to generate data for each submitted solution.  
The output data **must strictly follow the original dataset structure**:

- **Seismic data**: shape `(500, 5, 1000, 70)`
- **Velocity maps**: shape `(500, 1, 70, 70)`

Any submission that fails to conform to the format or system constraints may be disqualified from latency evaluation.



## License
This project is licensed under the BSD 3-Clause License. See the LICENSE file for full license text.

Portions of this project are based on the following open-source repositories:

- [InversionNet](https://github.com/lanl/OpenFWI), licensed under the BSD 3-Clause License
- [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), licensed under the MIT License
- [diffusion_distiller](https://github.com/Hramchenko/diffusion_distiller), licensed under the MIT License

We have retained all original license and copyright 
notices where applicable.