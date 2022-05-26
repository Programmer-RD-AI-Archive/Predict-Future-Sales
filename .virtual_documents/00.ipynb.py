import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import *
from torch.optim import *
from torchvision.models import *
from sklearn.model_selection import *
from sklearn.metrics import *
import wandb
import nltk
from nltk.stem.porter import *
PROJECT_NAME = "Natural-Language-Processing-with-Disaster-Tweets"
np.random.seed(55)
stemmer = PorterStemmer()
device = 'cuda'


item_categories = pd.read_csv("./data/item_categories.csv")
items = pd.read_csv("./data/items.csv")
data = pd.read_csv("./data/sales_train.csv")
sample_submission = pd.read_csv("./data/sample_submission.csv")
shops = pd.read_csv("./data/shops.csv")
test = pd.read_csv("./data/test.csv")


shops


item_categories


items


from tqdm import tqdm


data = {
    "Date":[],
    "Shop Id":[],
    "Item Id":[],
}


for i in tqdm(range(1)):
    d_iter = data.iloc[i]
    shop_iter = shops.iloc[shops['shop_id'].iloc[data['shop_id'][0]]]
    item_iter = items.iloc[items['item_id'].iloc[data['item_id'][0]]]
    item_cat_iter = item_categories.iloc[item_iter['item_category_id']]


d_iter


shop_iter


item_iter


item_cat_iter



