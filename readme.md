# The Tags Are Alright: Robust Large-Scale RFID Clone Detection Through Federated Data-Augmented Radio Fingerprinting  
## Authors: Mauro Piva<sup>1</sup>, Gaia Maselli<sup>1</sup>, Francesco Restuccia<sup>2</sup>  
### <sup>1</sup>Department of Computer Science, Sapienza University, Italy
### <sup>2</sup>Department of Electrical and Computer Engineering, Northeastern University, United States  
  
### Arxiv link: *****  

## Dataset
Our dataset, containing 200GB of raw RFID RN16 IQ Samples, is available at [Google Drive](https://drive.google.com/drive/folders/1l7cvR1Z8V7eiVdqcPgv0i5rb2ZMSja63?usp=sharing).

We suggest to use [GDrive](https://github.com/prasmussen/gdrive) or other similar software to speed up the entire dataset download.

Please notice that the dataset effective names are slightly different from their definition on our paper.  
  
### tags are encoded as:  

 - 000A -> 000 from OTA20  
 - 000B -> 000 from OTA50  
 - 000C -> 000 from OTA100   
 -  300A -> 000 from PM0 - 20  
 -  300B -> 000 from PM0 - 50  
 -  400A -> 000 from PM1 - 20  
 -  400B -> 000 from PM1 - 50

  
The dataset contains the following folders:   
  
 - **raw**, which contains the raw IQ samples for all the collected tags;   
 - **preprocessed**, which contains the IQ samples already divided into train and test and sliced selecting only the RN16.  

## Colab Fast Start

We are also releasing a notebook to fast start any test on our code.    
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mauropv/RFID-Fingerprint2020/blob/master/RFID_RFP_2020.ipynb)
Please notice that:   
  
1. Colab will not be a good option to run complex scenarios (e.g. data augmentation with 200 tags) , due to RAM size.  
   
2. Results reported in our paper have been obtained using a Tesla V100-DGXS.   
  

## Code Repository

This code is used to demonstrate our recent paper and to offer some example on ho to manage the related dataset.  
  
We used conda, and released an environment.yml showing packages used. To bootstrap the setup, install conda and run

    conda env create -f environment.yml  
    conda activate pyRFID-fingerprint

The file rfp-batch.py contains a train and test batch generator. You can update the last lines of that file to set up yout tests, for example defining the dataset to use, the number of federated clients, and if to use data augmentation.
  


  


