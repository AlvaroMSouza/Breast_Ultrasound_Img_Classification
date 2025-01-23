import kaggle

kaggle.api.authenticate()

# Download latest version
kaggle.api.dataset_download_files("aryashah2k/breast-ultrasound-images-dataset", path='.', unzip=True)
