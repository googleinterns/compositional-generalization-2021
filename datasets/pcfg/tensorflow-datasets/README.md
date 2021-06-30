## Creating TensorFlow datasets for PCFG

To create TensorFlow datasets for each split of the PCFG data (iid, productivity and systematicity, either in the original or in the iterative decoding form), 
first install the TFDS CLI tool:

    $ pip install -q tfds-nightly

Then, run the following command to build the TensorFlow dataset for, e.g., the productivity split of the original PCFG dataset:

    $ tfds build pcfg_productivity_data_original.py
    
Instructions for downloading/generating and storing the data needed to build each TensorFlow dataset can be found in the MANUAL_DOWNLOAD_INSTRUCTIONS of the 
corresponding tfds class.
