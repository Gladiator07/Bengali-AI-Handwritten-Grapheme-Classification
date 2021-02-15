#!/bin/bash


echo "This script will set your kaggle api-key to Kaggle api, download data
and arrange it in train, val, test directories"


# Put your Kaggle api key path here
echo "Fetching your Kaggle API Key"
kaggle_api_key_path='/content/drive/MyDrive/Kaggle/kaggle.json'

# This snippet will install kaggle api and connect your api-key to it
echo "Installing Kaggle API..."
pip3 install kaggle --upgrade
mkdir -p ~/.kaggle
echo "Setting up your Kaggle key to API..."
cp /content/drive/MyDrive/Kaggle/kaggle.json ~/.kaggle/
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle API Key successfully linked !!!"

# This snippet will download the data in root folder
echo "Downloading Data at root directory..."
cd ~/.
mkdir input
cd input/
kaggle competitions download -c bengaliai-cv19
echo "Unzipping *.parquet files"
unzip \*.zip

echo "Deleting *.zip files"
find . -name "*.zip" -type f -delete
echo "Data downloaded and unzipped successfully..."


echo "Installing dependencies"
pip3 install iterative-stratification    # for MultiLabelStratifiedKfold
 