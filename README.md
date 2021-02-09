# IMDB
> The purpose of the project was to create a neural network to classify movie reviews from IMDB dataset. Each movie review was either positive or negative.

## Sample review from IMDB

"I'm a Petty Officer 1st Class (E-6) and have been in the USCG for 6 years and feel that this movie strongly represents the Coast Guard. There were only a few scenes that were far fetched. The most far-fetched was when PO Fischer (Kutcher) went down inside of the sinking vessel to pull the vessel's captain out of the engine room... that would never happen. Swimmers are not allowed to go inside of any vessel no matter the circumstances. Second, the Command Center (supposedly in Kodiak), it looked more like a NASA command center... we don't have any gear that hi-tech. Third, the Captain of the Airstation would not be running the search & rescue cases with like 10 people on watch. In reality it would be an E-6 or E-7 as the SAR Controller and maybe 2 other support personnel like an assist SAR Controller & a Radio Watchstander. Otherwise the movie was dead on, I think they should have incorporated more of the other rates in the CG and their roles in search & rescue instead of just Aviation based rates. Some of the scenes from "A" school reminded me of my days their and the dumb stuff I did and got in trouble for in my younger days."

## Data

The "dataprocessing.py" file contains scripts that were used to convert raw data from IMDB into a format that can be used for building a model.  

## Model

The script for building the model is in lstm.py.

## Technologies
* Python - version 3.8.3
* Numpy - version 1.18.5
* Pandas - version 1.0.5
* PyTorch - version 1.7.0 CUDA 10.1
* PyTorch-Lightning - version 1.0.6
