# Ensemble of LSTMs and feature selection for human action prediction
This is a pytorch implementation of the paper "Ensemble of LSTMs and feature selection for human action prediction" submitted to IAS-16 conference.
The results published in the paper (http://arxiv.org/abs/2101.05645) have been obtained in MATLAB.

# WIP
- [x] Process the MoGaze dataset
- [ ] Train models proposed in the paper
- [ ] Test models proposed in the paper

# How to run
Download the MoGaze dataset:
```
mkdir data
cd data
wget https://ipvs.informatik.uni-stuttgart.de/mlr/philipp/mogaze/mogaze.zip
unzip mogaze
```

We have provided the humoro repository needed for extracting the data (version Dec 8 2020,  543dea9). If you want to install it from source, or have missing dependencies, do as in: https://humans-to-robots-motion.github.io/mogaze/getting_started

```
cd ../src
python3 -m pip install --upgrade pip --user
sudo apt install qt5-default
git clone https://github.com/PhilippJKratzer/humoro.git
cd humoro
python3 -m pip install -r requirements.txt --user
sudo python3 setup.py install
```

Install the requirements of our project:

`python3 -m pip install -r requirements.txt --user`

Note that you might need to manually install pytorch as in: https://pytorch.org/get-started/locally/

To extract MoGaze data needed for training and testing of our model please run (from the src folder):

`python3 extract_data.py`

`python3 split_data.py` 

Example of labeled data in /data/processed:

![Labeled Data](doc/processed.png "Labeled Data")
