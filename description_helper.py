import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import time
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

import string
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

model_name = "bert-base-nli-mean-tokens"
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_name)


def process_raw_desc(text):
    # split into words
    text = text.replace('-', ' ')
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    return ' '.join([word for word in stripped if word.isalpha()])


def create_sim_desc(products, engine):
    products_list = products.handle.dropna().unique()
    
    print('Started scraping products..\nwill take some time, please wait')
    products_df = []
    for product in tqdm(products_list):
        base_url = 'https://leaclothingco.com/products/'
        url = base_url + product
        try:
            response = requests.get(url)
    #         print(response.text)
            soup = bs(response.content, "html.parser")
            rev_div = soup.find("div",attrs={"class","Rte"}).get_text(strip=True)
            products_df.append((product, rev_div))
            time.sleep(1)
        except Exception as e:
            print(e,product)
    
    print(f'Products scraped: {len(products_df)}')

    desc = pd.DataFrame(products_df, columns=['Handle','Description'])

    #dump this file if neccessary
    # products_desc.to_csv('Product_Descriptions.csv', index = False)

    #process data to be converted into required forms
    desc.Description = desc.Description.map(process_raw_desc)

    sentence_vecs = model.encode(desc.Description)

    sim_dict = dict()
    for i in range(sentence_vecs.shape[0]):
        res = cosine_similarity( [sentence_vecs[i]] , sentence_vecs )[0]
        indices = (-res).argsort()[1:10]
        res_list = []
        for index in indices:
            res_list.append(desc.iloc[index,0])
        sim_dict[ desc.iloc[i,0] ] = ','.join(res_list)
    
    # dump if required in pickle format
    # pickle.dump( sim_dict, open('sim_desc', 'wb'))

    sim_desc = pd.Series(sim_dict, index = sim_dict.keys()).reset_index().rename(columns = {'index':'product',0:'sim_products'})

    sim_desc.to_sql('sim_desc', engine, index = False, if_exists = 'replace' )
    print('SIM_DESC successfully created..')



