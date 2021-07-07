## Creating TensorFlow datasets for PCFG

To create TensorFlow datasets for each split of the PCFG data (iid, productivity and systematicity), first install the TFDS CLI tool:

    $ pip install -q tfds-nightly

Then, run the following command to build the TensorFlow dataset for, e.g., the productivity split of the PCFG dataset:

    $ tfds build pcfg_productivity_data.py --manual_dir="../"
    
The argument manual_dir specifies the location of the "data/" folder. Instructions for downloading/generating and storing the data needed to build each TensorFlow dataset can be found in the MANUAL_DOWNLOAD_INSTRUCTIONS of the corresponding tfds class.
