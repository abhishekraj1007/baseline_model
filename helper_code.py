import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import random
from random import choices
from nltk.tokenize import word_tokenize
import string
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
base_url = 'https://leaclothingco.com/products/'


def process_product_name(string):
    #splitting to get original string
    string = string.split(' - ')[0].strip().split()
    new_string = []
    for part in string:
        if not part[0].isalnum():
            if not part[-1].isalnum():
                res = part[1:-1]
            else:
                res = part[1:]
        elif not part[-1].isalnum():
            res = part[:-1]
        else:
            res = part
        if res!='':
            new_string.append(res.lower())
    return '-'.join(new_string)

# process_product_name('REINA BLACK MESH + LACE BUSTIER CORSET TOP - Regular/Xs')


def pre_process(filename = 'orders_export_1.csv'):
    #Read Orders file
    orders = pd.read_csv(filename)

    #changing column names for convenience
    cols = orders.columns
    cols = [''.join(item.title() for item in entity.split()) for entity in cols]
    orders.columns = cols

    #mapping missing and improper data
    map1 = orders.dropna(subset=['FinancialStatus'])[['Name','FinancialStatus']]
    orders = pd.merge(orders, map1, on='Name', suffixes=('_old', ''))
    map2 = orders.dropna(subset=['FulfillmentStatus'])[['Name','FulfillmentStatus']]
    orders = pd.merge(orders, map2, on='Name', suffixes=('_old', ''))

    # Selecting required columns
    req = ['Name', 'Email','LineitemName','LineitemFulfillmentStatus','FinancialStatus','FulfillmentStatus']
    orders = orders[req]


    ## creating product list for filtering
    products = pd.read_csv('products_export.csv')

    products_list = products.Handle.dropna().unique()
    print(len(products_list))

    # finally creating mappings
    product2id = dict( (b,a) for a,b in enumerate(products_list) )
    id2product = dict(enumerate(products_list))
    
        
    #processing product names(Handle- Title conflicts)
    orders.LineitemName = orders.LineitemName.apply(process_product_name)
    
    # correcting some errors in data
    temp_mappings = {'carla-mauve-silk-corset-top':'carla-mauve-silk-mesh-corset-top',
                 'twyla-mesh-corset-t-shirt':'twyla-black-mesh-corset-t-shirt',
                'tatiana-ruched-midi-corset-dress':'tatiana-red-ruched-mesh-midi-corset-dress',
                'belle-lavender-ombre-ruffle-tulle-corset-dress':'belle-lavender-ombro-ruffle-tulle-corset-dress',
                'brielle-teddy-wide-leg-pants':'brielle-lavender-teddy-wide-leg-pants',
                'betty-teddy-long-cardigan':'betty-lavender-teddy-long-cardigan',
                'brie-teddy-crop-top':'brie-lavender-teddy-crop-top',
                'aphrodite-embroidered-gown-skirt':'aphrodite-embroidered-sheer-gown-skirt'}
    
    old_mappings = dict(zip(orders.LineitemName, orders.LineitemName))
    for key,value in temp_mappings.items():
        old_mappings[key] = value
    orders.LineitemName = orders.LineitemName.map(old_mappings)
    
    #removing outdated products
    orders = orders.loc[orders.LineitemName.isin(products_list)]
    
    #selecting carts with two or more products
    order_counts = orders.Name.value_counts()
    counts_index = order_counts[order_counts > 1].index

    # taking 2 or more cart values
    orders = orders.loc[orders.Name.isin(counts_index)]
    
    # converting products to their ids
    orders['ProductId'] = orders.LineitemName.map(product2id).astype(int)
    
    #scoring user interactions
    rate_dict = {'paid':5, 'pending':4, 'voided':3, 'refunded':1}
    orders['rating'] = orders.FinancialStatus.map(rate_dict)
    
    orders = orders[['Email', 'ProductId', 'rating']]
    orders.reset_index(drop=True, inplace=True)

    #preparing for missing ids in training data to avoid errors at inference
    missing_ids = list(set(range(len(products_list))) - set(orders.ProductId.unique()))
    for pid in missing_ids:
        orders.loc[len(orders.index)] = [str(pid) +'@Dummy', pid, 5]
    
    # pivoting tables for training data
    orders = orders.pivot_table('rating',['Email'],'ProductId')
    orders.fillna(0, inplace = True)
    
    #writing users(orders) to disk
    print('Processing finished, writing users to disk...\n')
    out_filename = 'users.csv'
    orders.to_csv(out_filename, index = True)
    return out_filename, id2product
    
    
def get_sim(filename = 'users.csv'):
    orders = pd.read_csv(filename, header = 0, index_col = 0)
    print(orders.isna().sum().sum())
    
    ids = list(orders.columns)
    idx_to_ids = dict(enumerate(ids))
    ids_to_idx = dict([(y,x) for x,y in idx_to_ids.items()])


    sim = np.zeros((len(ids),len(ids)))
    
    
    #Making correlation matrix using ADJUSTED COSINE SIMILARITY

    avg_item_ratings = orders.values.mean(axis = 1)
#     avg_item_ratings = np.true_divide(orders.sum(axis = 1),(orders!=0).sum(axis = 1))

    for row in tqdm(range(len(ids))):
        for col in range(len(ids)):
            if sim[row][col]!=0:
                continue
            #Using pearson similarity here (subtracting item's avg ratings)
            r_u_i = orders.loc[:,idx_to_ids[row]].values - avg_item_ratings
            r_u_j = orders.loc[:,idx_to_ids[col]].values - avg_item_ratings

            # Dot product or sum of product of arrays
            num = np.matmul(r_u_i,r_u_j)

            den_1 = np.sqrt(np.sum(np.square(r_u_i)))
            den_2 = np.sqrt(np.sum(np.square(r_u_j)))
            den = den_1 * den_2
            res = num/den
            sim[row][col] = res
            sim[col][row] = res
    
    sim = pd.DataFrame(sim, columns=ids, index = ids)
    print('Training Corr matrix finished...\nsaving weights...\n')
    sim.to_csv('sim.csv', index = True)
    
    
def get_inference(email, product_id, sim, users, avg_item_ratings, id2product, reco_count = 10):
        
    if email in users.index:
        print('Found existing user...')
#         print('Entered part 1\n')
        temp_user = users.loc[email]
        #Assume customer likes/Bought this product
        temp_user.loc[str(product_id)] = 5.0
        
        
        #Normalizing the rating scale by subtracting average ratings by users
        r_u_j = temp_user.values - avg_item_ratings.values

        ##Getting inference using:
        ## Score(u,i) = sum(sim(i,j)*(r(u,j) - r_avg(j)))/sum(sim(i,j)) + r_avg(i)
        ## Where summation is for all users i.e. from j to I.
        
        res = []
        for item in temp_user.index.values:
            #Skip items which are already bought or the item i.e. currently viewed
            if temp_user.loc[item]>=5 or item == product_id:
                continue

            sim_i = sim.loc[:,item].values
            num = np.matmul(sim_i, r_u_j)

            den = sim.loc[:,item].sum()

            score = (num/den) + avg_item_ratings[item]
            res.append([item,score])

        res.sort(key = lambda x: x[1], reverse = True)
        
        results = []
        for product,score in res[:reco_count]:
            prod_url = base_url + id2product[int(product)]
#             print(prod_url)
            results.append(prod_url)
        return results

    else:
        # Demographics Based
        #2a Explore User Profile
        print('Found new user...')
        try:
#             print('Entered part 2\n')
            cust = pd.read_csv('customers_export.csv')
            province = cust.loc[cust.Email == 'neelam.madnani98@gmail.co','Province'].values[0]
        except Exception as e:
            #Check the Demographics table for available user info
            province = get_demographics(email)
            
        if province != -1:
            part2 = set(get_demographic_recos(product_id, id2product, ShippingProvinceName = province, orders_filename = 'orders_export_1.csv'))
        else:
            print('user profile unavailable,.. sending popup..\n')
            part2 = set()
        
        
        #2b show content based recos
#         print('Entered part 3\n')
        part3 = set( list( map(id2product.get, [int(item) for item in sim.loc[product_id,:].nlargest(reco_count+1).index[1:]]) )  )
#         res = sim.loc[:,str(product_id)].sort_values(ascending = False)[1:reco_count+1]
        
    
    
        #2c get_similar descriptions based products
#         print('Entered part 4\n')
        part4 = set( get_similar_desc(product_id, mapping = id2product) )
#         print(part2, part3, part4, sep = '\n')
        
        # sampling recommendations based on priority based function
        # The three position for weights represent prob dist for selection from each arr
        Model_weights = [1, 1.5, 1]
        results = weighted_sample_without_replacement( arrs= [ part2, part3, part4 ], weight_each_arr = Model_weights, k=reco_count)
        return [base_url + item for item in results]
    
        
def weighted_sample_without_replacement(arrs, weight_each_arr, k=2):
    population = []
    weights = []
    for i,arr in enumerate(arrs):
        population.extend(arr)
        weights.extend( [ weight_each_arr[i] ] * len(arr) )
    
    weights = list(weights)
    positions = range(len(population))
    indices = []
    while True:
        needed = k - len(indices)
        if not needed:
            break
        for i in choices(positions, weights, k=needed):
            if weights[i]:
                weights[i] = 0.0
                indices.append(i)
    return [population[i] for i in indices]
            
            
def get_similar_desc(product_id, mapping):
    sim_desc = pickle.load(open('sim_desc','rb'))
    product_name = mapping[product_id]
    
    return sim_desc[product_name]
        
    
    
def get_demographic_recos(product_id, mapping, ShippingProvinceName = 'Delhi', orders_filename = 'orders_export_1.csv'):
    orders = pd.read_csv(orders_filename)
    
    df = pre_processing_orders(orders)
    
    return_dict = Item_purchased_Together_statewise(df,ShippingProvinceName)
    print(f'Demographic results for Location: {ShippingProvinceName} are included')
    
    product_name = mapping[product_id]
    res = return_dict.get( product_name, [] )[:5]
    if res:
        return [base_url + item for item in res]
    else:
        print('No demographics data for the product found in the Locality')
        return []
    
    
def get_demographics(email):
    
    # Needs to done through database
    return 'Delhi'
    
    
    
def pre_processing_orders(orders):
    orders = orders.copy()
    cols = orders.columns
    cols = [''.join(item.title() for item in entity.split()) for entity in cols]  
    orders.columns = cols
    orders.LineitemName = orders.LineitemName.map(process_product_name)

    return orders
    
    
    
def Item_purchased_Together_statewise(df,state):
    df = df[df['ShippingProvinceName']==state].reset_index()
    vc = df.LineitemName.value_counts()
    pairs = {}

    for j,i in enumerate(vc.index.values):
    # print(j,i)
        users = df.loc[df.LineitemName == i, 'Email'].unique()
        vc2 = df.loc[(df.Email.isin(users))&(df.LineitemName!=i),'LineitemName'].value_counts()
        pairs[i] = vc2.index.to_list()[:5]

    return pairs


def model_fn(orders_filename):
    users_filename, id2product = pre_process(orders_filename)
    
    get_sim(users_filename)
    sim = pd.read_csv('sim.csv', index_col = 0, header = 0)
    users = pd.read_csv('users.csv', index_col = 0, header = 0)
    
    avg_item_ratings = users.mean(axis = 0)
    
    base_url = 'https://leaclothingco.com/products/'
    
    return  sim, users, avg_item_ratings, id2product, base_url
