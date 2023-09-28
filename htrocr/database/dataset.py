import os
import sys
import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_url
#import pytesseract
import cv2

NHMD_DATA_URL = 'https://specify-attachments.science.ku.dk/static/NHMD_Botany/originals/'
DATA_ROOT = '../data'
NHMD_ORIG = 'NHMD_ORIG'
NHMD_ORIG_LIST_PATH = '../data/NHMD_ORIG/info.csv'
NHMD_BASELINE_TRAIN = 'NHMD_BASELINE_TRAIN'
NHMD_BBOX = 'NHMD_BBOX'
NHMD_DATASET_CSV = 'NHMD_dataset.csv'
SEL_DATASET_CSV = 'info.csv'
year_col = '1,10.collectingevent.startDateNumericYear'
catalog_col = '1.collectionobject.catalogNumber'
url_col = '1,111-collectionobjectattachments,41.attachment.attachmentLocation'


def download_images(selected_df, path, url=NHMD_DATA_URL, force_update=False):
    if (force_update or not os.path.exists(path)):
        os.makedirs(path)
        selected_df.to_csv(path + SEL_DATASET_CSV)
        imgs = selected_df[url_col]
        for img in imgs:
            download_url(url+img, path)

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

def year_stratified_split(dataset, lower_bound=1799, upper_bound=1950, group_size=10, elements_to_take=20, seed=None):
    if seed is not None:
        np.random.seed(seed)

    groups = dataset.groupby(pd.cut(dataset[year_col], np.arange(lower_bound, upper_bound, group_size))).groups
    selected_data = []
    for group in groups:
        group_arr = groups[group].to_numpy()
        np.random.shuffle(group_arr)
        random_samples = group_arr[0:elements_to_take]
        selected_data = selected_data + list(random_samples)
    return dataset.filter(items = selected_data, axis=0)


def get_sorted_contours(img):
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours)==2 else contours[1]
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    return contours

def add_bbox(imageset_df, in_path, out_path):
    img_names = imageset_df[url_col]
    img_name = 'sp622660014736117175.att.jpg'
    img_to_read = os.path.join(in_path, img_name)
    image = cv2.imread(img_to_read)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)    
    (_,thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    struct_el = cv2.getStructuringElement(cv2.MORPH_RECT, (60,60))
    dilated = cv2.dilate(thresh, struct_el, iterations=1)
    if (not os.path.exists(out_path)):
        os.makedirs(out_path)
    cv2.imwrite(out_path+'/dilated.png', dilated)
    contours = get_sorted_contours(dilated)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if h < 2500 and h > 100:
            cv2.rectangle(image, (x,y), (x+w, y+h),(0, 0, 255), 2)
    cv2.imwrite(out_path+'/bboxed.png', image)
                
def gen_test_dataset(force_update):
    '''
    Generator for HTR test samples
    '''
    sel_img_data_path = os.path.join(DATA_ROOT, NHMD_ORIG)
    sel_img_df_path = os.path.join(sel_img_data_path, SEL_DATASET_CSV)
    selected_df = None
    orig_df, valid_df = read_and_clean_dataset()
    analyse_dataset(orig_df, valid_df)
    # Check if images were already downloaded
    if (force_update==True or not os.path.exists(sel_img_df_path)):
        orig_df, valid_df = read_and_clean_dataset()
        analyse_dataset(orig_df, valid_df)
        selected_df = year_stratified_split(valid_df)
        download_images(selected_df, sel_img_data_path)
    else:
        selected_df = pd.read_csv(sel_img_df_path)
    # Extract text regions
    bbox_img_path = os.path.join(DATA_ROOT, NHMD_BBOX) 
    add_bbox(selected_df, sel_img_data_path, bbox_img_path)


def pull_baseline_train_db():
    '''
    Generator for ARU-NET train db samples. Samples still need to be pre-processed into separate
    channels. Next step is to extract baselines - one way is to use Transkribus
    '''
    selected_df = None
    # Read test db csv
    _, test_df = read_and_clean_dataset(NHMD_ORIG_LIST_PATH)
    sel_img_data_path = os.path.join(DATA_ROOT, NHMD_BASELINE_TRAIN)
    sel_img_df_path = os.path.join(sel_img_data_path, SEL_DATASET_CSV)
    if (not os.path.exists(sel_img_df_path)):
        # Read full db csv
        _, valid_df = read_and_clean_dataset()
        selected_df = year_stratified_split(valid_df, elements_to_take=110)
        analyse_dataset(selected_df, selected_df)
        cond = selected_df[url_col].isin(test_df[url_col])
        selected_df.drop(selected_df[cond].index, inplace = True)
        print(f"Total number of training samples: {selected_df.shape[0]}")
        download_images(selected_df, sel_img_data_path)



if __name__ == '__main__':
    # gen_test_dataset(sys.argv[1])
    gen_baseline_train_db()
