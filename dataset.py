import os
import numpy as np
import pandas as pd
import PIL.Image as Image
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, DataLoader, Subset

NHMD_DATA_URL = 'https://specify-attachments.science.ku.dk/static/NHMD_Botany/originals/'
DATA_ROOT = './data'
NHMD_DATASET_CSV = './NHMD_dataset.csv'
year_col = '1,10.collectingevent.startDateNumericYear'
catalog_col = '1.collectionobject.catalogNumber'
url_col = '1,111-collectionobjectattachments,41.attachment.attachmentLocation'


def download_images(selected_df, url=NHMD_DATA_URL, force_update=False, location='NHMD_ORIG', root=DATA_ROOT):
    data_folder = os.path.join(root, location)
    if (force_update or not os.path.exists(data_folder)):
        os.makedirs(data_folder)
        selected_df.to_csv(data_folder + '/info.csv')
        imgs = selected_df[url_col]
        for img in imgs:
            download_url(url+img, data_folder)
    os.listdir(data_folder)

def read_and_clean_dataset(csv_path=NHMD_DATASET_CSV):
    df = pd.read_csv(csv_path)
    # Remove outliers
    validSamples = df[(df[year_col] > 1000) & df[url_col].notna()]
    return df, validSamples

def analyse_dataset(df, validSamples):
    years = np.sort(df[year_col].unique())
    years_valid = np.sort(validSamples[year_col].unique())        
    print("CSV Stats:")
    print(f"Total number of samples: {df.shape[0]}")
    print(f"Total year range: {years[0]} - {years[years.size - 1]}")
    print(f"Number of samples with images: {validSamples.shape[0]}")
    print(f"Valid image year range: {years_valid[0]} - {years_valid[years_valid.size - 1]}")
    print(f"Records per year (valid data): \n{validSamples[year_col].value_counts().sort_index()}")
    print(f"Duration (in years) of valid sample collection: {years_valid[years_valid.size - 1] - years_valid[0]}")
    print(f"# of valid records before 1900: {validSamples[validSamples[year_col]<1900].shape[0]}")
    print(f"# of valid records after 1900: {validSamples[validSamples[year_col]>=1900].shape[0]}")
    prev = 0
    record_history = []
    for cur in range(10, 150, 10):
        cond1 = validSamples[year_col]>=years_valid[0]+prev
        cond2 = validSamples[year_col]<years_valid[0]+cur
        range_x = validSamples[cond1&cond2].shape[0]
        record_history.append({f'{years_valid[0]+prev}-{years_valid[0]+cur}':range_x})
        print(f"# of valid records between {years_valid[0]+prev} and {years_valid[0]+cur} : {range_x}")
        prev = cur

def year_stratified_split(dataset, lower_bound=1799, upper_bound=1950, group_size=10, seed=None):
    if seed is not None:
        np.random.seed(seed)

    groups = dataset.groupby(pd.cut(dataset[year_col], np.arange(lower_bound, upper_bound, group_size))).groups
    selected_data = []
    for group in groups:
        group_arr = groups[group].to_numpy()
        np.random.shuffle(group_arr)
        random_samples = group_arr[0:20]
        selected_data = selected_data + list(random_samples)
    return dataset.filter(items = selected_data, axis=0)

if __name__ == '__main__':
    orig_df, valid_df = read_and_clean_dataset()
    # analyse_dataset(orig_df, valid_df)
    selected_df = year_stratified_split(valid_df)
    download_images(selected_df)
    

