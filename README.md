# Magmaw
Official implementation for *Magmaw: Modality-Agnostic Adversarial Attacks on
Machine Learning-Based Wireless Communication Systems*. 

We validate Magmaw through simulation, and then thoroughly conduct a real-world evaluation with [Software-Defined Radio (SDR)](https://wiki.gnuradio.org/index.php/Tutorials).

## Prerequisite
Magmaw is implemented with [Python 3.7](https://www.python.org/downloads/) and [PyTorch 1.7.1](https://pytorch.org/). We manage the development environment using [Conda](https://anaconda.org/anaconda/conda).

Please go ahead and execute the following commands to configure the development environment.
- Create a conda environment called `Magmaw` based on Python 3.7, and activate the environment.
    ```bash
    conda create -n Magmaw python=3.7 --file requirements.txt
    conda activate Magmaw
    git clone hhttps://github.com/Magmaw/Magmaw.git
    ```

## Dataset

### - Test Datasets for Image JSCC and JSCC
We evaluate the `image JSCC` and `video JSCC` models using the UCF-101 dataset from this [repo](https://github.com/sli057/Geo-TRAP).

### - Test Datasets for Speech JSCC and Text JSCC
We evaluate the `speech JSCC` using Edinburgh DataShare. 

We select the proceedings of the European Parliament to evaluate the `text JSCC`.


These datasets can be downloaded [here](https://drive.google.com/drive/folders/1pxZ9pdtlIz3KdNd-M_uDSD0p8UzjG_BE?usp=sharing).

### - Path Configuration
Please edit the paths for the dataset in `configs/config.py`.

## Checkpoints
The checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1mKj0EK5hC1EATHaD0RfSvVGm5WbEE6R0?usp=drive_link).

Please edit the paths for checkpoints in `configs/config.py`.


## Usage for Simulation Results

To run the black-box attack on the multimodal JSCC models:

```bash
python black_box_attack.py
```

## Simulation Results

```
psr: -16 cd_rate: 1 mod: QAM degree: 16
Attack: random, Morality video, RX psnr : 32.92, RX msssim: 0.909
Attack: black, Morality video, RX psnr : 27.72, RX msssim: 0.805
Attack: random, Morality image, RX psnr : 34.08, RX msssim: 0.925
Attack: black, Morality image, RX psnr : 28.10, RX msssim: 0.823
Attack: random, Morality speech, RX MSE: 0.0000550
Attack: black, Morality speech, RX MSE: 0.0006523
Attack: random, Morality text, RX BLEU_1g: [0.92747291]
Attack: random, Morality text, RX BLEU_2g: [0.8617476]
Attack: random, Morality text, RX BLEU_3g: [0.79970497]
Attack: random, Morality text, RX BLEU_4g: [0.73994322] 
Attack: black, Morality text, RX BLEU_1g: [0.48864823]
Attack: black, Morality text, RX BLEU_2g: [0.2488876]
Attack: black, Morality text, RX BLEU_3g: [0.13983744]
Attack: black, Morality text, RX BLEU_4g: [0.08387463]
```

## SDR Implementation
<div align="center"> <img src="./SDR_setup.png" height=250> </div>

The block diagram of our SDR implementation is presented in the above figure. Following the above block diagram, we construct the legitimate transmitter, legitimate receiver, and adversarial transmitter. 

We utilize [GNURadio software package](https://wiki.gnuradio.org/index.php/Tutorials) to control USRP SDRs.

We follow the steps below. 

* We first store the symbols encoded by multimodal JSCC in a txt file.

* Then, we feed the stored OFDM symbols to the OFDM transmitter to send the radio signal over the air.

* The OFDM receiver converts the received signals into complex-valued symbols, and the JSCC decoder restores them to the original data.

* Note that the power of signal transmission is controlled by adjusting the signal amplitude during the signal generation process.

<div align="center"> <img src="./fig_real_scenario.png" height=400> </div>

We show one of the real-world attack scenarios in the above Figure. 

## Real-World Data

Using GNURadio, we stored the index of constellation points generated after demodulation.

We also store the reference data to compare the results.

```
Magmaw
├── SDR_results
    ├── black 
    │   ├── image
    |   │       ├── ori.txt   /* original input */
    |   |       ├── ref.txt   /* simulated output from JSCC */
    |   |       ├── wir.txt   /* real-world output */
    ├── no
    │   ├── image
    |   │       ├── ori.txt   /* original input */
    |   |       ├── ref.txt   /* simulated output from JSCC*/
    |   |       ├── wir.txt   /* real-world output */
```

## Usage for SDR results

To send the real-world data to the JSCC models:

```bash
python evaluate_SDR.py
```

## SDR Results

```
Validation, Attack: no, Morality: image, TX psnr : 34.37, TX msssim: 0.940, RX psnr : 33.30, RX msssim: 0.927
Validation, Attack: black, Morality: image, TX psnr : 34.37, TX msssim: 0.940, RX psnr : 25.81, RX msssim: 0.724
```
