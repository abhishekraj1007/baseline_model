##Things to test: Test process_product_name for correct conversions, check if the style matching criteria is correct.

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from random import choices
# from nltk.tokenize import word_tokenize
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pymongo
base_url = 'https://leaclothingco.com/products/'

db_name = 'lea_clothing_backend'
collection_name = 'processed_user_profiles'


def pre_process(orders_filename = 'orders_export_1.csv', products_filename = 'products_export_1.csv'):
    #Read Orders file
    orders = pd.read_csv(orders_filename)

    #changing column names for convenience
    cols = orders.columns
    cols = [''.join(item.title() for item in entity.split()) for entity in cols]
    orders.columns = cols

    #mapping missing and improper data
    map1 = orders.dropna(subset=['FinancialStatus'])[['Name','FinancialStatus']]
    orders = pd.merge(orders, map1, on='Name', suffixes=('_old', ''))
    map2 = orders.dropna(subset=['FulfillmentStatus'])[['Name','FulfillmentStatus']]
    orders = pd.merge(orders, map2, on='Name', suffixes=('_old', ''))
    

    products = pd.read_csv(products_filename)
    #dropping duplicates
    products.drop_duplicates(subset='Title', keep="first", inplace= True)

    products = products[['Handle','Title']]
    products.dropna(subset=['Title'], inplace = True)
    products.reset_index(inplace=True, drop = True)

    # finally creating mappings title -> handle and handle -> title
    handle2title = dict(zip(products.Handle, products.Title))
    title2handle = dict(zip(products.Title, products.Handle))

    ## creating product list for filtering
    products_title_list = products.Title.dropna().unique()
    products_handle_list = products.Handle.dropna().unique()
        
    #processing product names(Handle- Title conflicts)
    orders.LineitemName = orders.LineitemName.apply(lambda x: x.split(' - ')[0].strip())
    
    #removing outdated products
    orders = orders.loc[orders.LineitemName.isin(products_title_list)]
    
    # converting product Titles to their Handles
    orders.LineitemName = orders.LineitemName.map(title2handle)
    
    #saving orders for further use
    orders.to_csv('orders_export_processed.csv', index = False)
    
    # Selecting required columns
    req = ['Name', 'Email','LineitemName','LineitemFulfillmentStatus','FinancialStatus','FulfillmentStatus']
    orders = orders[req]
    
    #selecting carts with two or more products
    order_counts = orders.Name.value_counts()
    counts_index = order_counts[order_counts > 1].index

    # taking 2 or more cart values
    orders = orders.loc[orders.Name.isin(counts_index)]
    
    # converting product Titles to their Handles
    orders['ProductHandle'] = orders.LineitemName
    
    #scoring user interactions
    rate_dict = {'paid':5, 'pending':4, 'voided':3, 'refunded':1}
    orders['rating'] = orders.FinancialStatus.map(rate_dict)
    
    orders = orders[['Email', 'ProductHandle', 'rating']]
    orders.reset_index(drop=True, inplace=True)

    #preparing for missing products in training data to avoid errors at inference
    missing_ids = list(set(products_handle_list) - set(orders.ProductHandle.unique()))
    for pid in missing_ids:
        orders.loc[len(orders.index)] = [str(pid) +'@Dummy', pid, 5]
    
    # pivoting tables for training data
    orders = orders.pivot_table('rating',['Email'],'ProductHandle')
    orders.fillna(0, inplace = True)
    
    #writing users(orders) to disk
    print('Processing finished, writing users to disk...\n')
    out_filename = 'users.csv'
    orders.to_csv(out_filename, index = True)
    return out_filename, title2handle    
    
def get_sim(filename = 'users.csv'):
    orders = pd.read_csv(filename, header = 0, index_col = 0)
    
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
    
    
def get_inference(email, product_title, sim, users, avg_item_ratings, title2handle, tag_array, connection, reco_count = 10):
    
    product_handle = title2handle[product_title]
    
    if email in users.index:
        print('Found existing user...')
#         print('Entered part 1\n')
        temp_user = users.loc[email]
        #Assume customer likes/Bought this product
        temp_user.loc[product_handle] = 5.0
        
        #Normalizing the rating scale by subtracting average ratings by users
        r_u_j = temp_user.values - avg_item_ratings.values

        ##Getting inference using:
        ## Score(u,i) = sum(sim(i,j)*(r(u,j) - r_avg(j)))/sum(sim(i,j)) + r_avg(i)
        ## Where summation is for all users i.e. from j to I.
        
        res = []
        for item in temp_user.index.values:
            #Skip items which are already bought or the item i.e. currently viewed
            if temp_user.loc[item]>=5 or item == product_handle:
                continue

            sim_i = sim.loc[:,item].values
            num = np.matmul(sim_i, r_u_j)

            den = sim.loc[:,item].sum()

            score = (num/den) + avg_item_ratings[item]
            res.append([item,score])

        res.sort(key = lambda x: x[1], reverse = True)
        
        results = []
        for product,score in res[:reco_count]:
            prod_url = base_url + product
#             print(prod_url)
            results.append(prod_url)
        return results

    else:
        # Demographics Based
        #2a Explore User Profile
        print('Customer order history not found...')
        try:
            cust = pd.read_csv('customers_export.csv')
            province = cust.loc[cust.Email == email,'Province'].values[0]
            part2 = get_demographic_recos(product_handle, ShippingProvinceName = province, orders_filename = 'orders_export_processed.csv')
        except Exception as e:
            part2 = []
        
#         2b show content based recos
#         print( sim.loc[product_handle,:].nlargest(reco_count+1).index[1:])
        part3 = sim.loc[:,product_handle].sort_values(ascending = False)[1:reco_count+1].index.to_list()
        
        
        #2c get_similar descriptions based products
        part4 = get_similar_desc(product_handle)
        
        #2d get_tag_based_personalized recommendations else show similar tag based products
        #Pending, to be done through Database
        try:
            #fetching user attributes from mongodb
            tag_profile = get_user_tag_profile(connection, email, tag_array.columns)
            print(f'Found tag profile: {email}')
            part5 = get_tag_based_inference(tag_profile , tag_array , title2handle, standalone = True, n_recos = 15)
            
        except Exception as e:
            print(f'User tag profile not found, {e}\n')
            #pending
            tag_profile = tag_array.loc[tag_array.index == product_handle ].iloc[-1,:]
            part5 = get_tag_based_inference(tag_profile, tag_array , title2handle, standalone = True, n_recos = 15)[1:]
    
#         #print output and verify results
#         print(f'Part2: {part2},\n Part3:{part3},\n Part4:{part4},\n Part5:{part5}\n')
        
        # sampling recommendations based on priority based function
        # The three position for weights represent prob dist for selection from each arr
        Model_weights = [1, 1.5, 1, 1]
        results = weighted_sample_without_replacement( arrs= [ part2, part3, part4, part5 ], weight_each_arr = Model_weights, k=reco_count)
        return results
        
        
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
            
            
def get_similar_desc(product_handle):
    sim_desc = pickle.load(open('sim_desc','rb'))
    return sim_desc[product_handle]
        
    
    
def get_demographic_recos(product_handle, ShippingProvinceName = 'Delhi', orders_filename = 'orders_export_processed.csv'):
    df = pd.read_csv(orders_filename)
    
    return_dict = Item_purchased_Together_statewise(df,ShippingProvinceName)
    print(f'Demographic results for Location: {ShippingProvinceName} are included')
    
    res = return_dict.get( product_handle, [] )[:5]
    if res:
        return res
    else:
        print('No demographics data for the product found in the Locality')
        return []  
    
    
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

    
def model_fn(orders_filename, products_filename, tags_filename):
    users_filename, title2handle = pre_process(orders_filename, products_filename)
    
    get_sim(users_filename)
    sim = pd.read_csv('sim.csv', index_col = 0, header = 0)
    users = pd.read_csv('users.csv', index_col = 0, header = 0)
    
    avg_item_ratings = users.mean(axis = 0)
    
    base_url = 'https://leaclothingco.com/products/'
    try:
        productsXtags = pd.read_csv(tags_filename, header = 0, index_col = 0)
    except Exception as e:
        print(f'Products tag file cant be read: {e}')
        productsXtags = -1

    return  sim, users, avg_item_ratings, title2handle, base_url, productsXtags


###############################################################
###############################################################
#API 2


def create_product_tags_arr(filename = 'products_export_1.csv'):
    """
    To create products X tags array to be used for matrix multiplication for cosine similarity afterwards
    """
    products = pd.read_csv(filename)
    
    #dropping duplicates
    products.drop_duplicates(subset='Title', keep="first", inplace= True)
    products.dropna(subset=['Title','Tags'], inplace = True)
    products.reset_index(inplace=True, drop = True)
    title2handle = dict(zip(products.Title, products.Handle))
    print(products.shape)
    
    products = products[['Handle','Tags']]

    products.Tags = products.Tags.apply(lambda x: x.split(', '))
    products = products.explode('Tags')
    products.Tags = products.Tags.apply(lambda x: ''.join( item.lower() for item in x.replace('-','').split() ))
    products['count'] = 1

    products = products.pivot_table('count', ['Handle'],'Tags')
    products.fillna(0, inplace = True)
    
    processed_filename = 'productsXtags.csv'
    products.to_csv(processed_filename, header = True, index = True)
    return processed_filename, title2handle


def create_profile(data, productsXtags_arr):
    """
    function to process raw json into useful user profile
    """
    
    # a slightly smaller weight is assigned to filter tags
    uncomfortable_dict = {'Arms':{'Sleeves':1}, 'Waist':{'High Waist':1, 'Skater':1, 'Shift':1, 'Slip':1, "Bodycon":-2, "Crop Top":-2},
                         'Legs': {'Midi':1, 'Gown':1, 'Pants':1, 'Maxi':1, 'Gown':1},
                         'Back':{'Backless':-2},
                         'Collarbones':{"Off-Shoulder":-2, "Strapless":-2}}
    
    #Questions and corresponding weights of thier tags
    body_tags = {'Bodies':2, 'accentuate':4, 'uncomfortable': 4, 'height':1, 'colour palettes':2,
                'prints':2,'occasion':5 }
    
    tags = dict()
    for item,weight in body_tags.items():
        if item!='uncomfortable':
            values = data.get(item, [])
            if values:
                values = values['value']
            
            if isinstance(values, list):
                temp = []
                for value in values:
                    temp.extend(value.split(', '))
            else:
                temp = [values]
                
#             print('temp',temp)
            tags.update(dict( (
                ''.join( item.lower() for item in tag.replace('-','').split() ),weight) for tag in temp ))
            
#             print(tags)
        else:
            values = data.get(item, [])
            if values:
                values = values['value']
                
            for tag in values:
                value = uncomfortable_dict[tag]
                
                tags.update(dict( (
                    ''.join( item.lower() for item in key.replace('-','').split() ), value * weight) for key,value in value.items() ))
#             print('uncomfortable', values, tags)
    tag_profile = pd.Series(tags, index = list(productsXtags_arr.columns))
    tag_profile.fillna(0, inplace= True)
    return tag_profile


def get_tags_sim(tag_profile, tag_array, n):
    """
    Function to fetch n most similar indices based on 1d array and another 2d array
    """
    
    #notice that order has been reversed
    ids = (-cosine_similarity( tag_array, [tag_profile])).argsort(axis = 0)[:n].flatten()
    
#     print(cosine_similarity( tag_array, [tag_profile]) , cosine_similarity( tag_array, [tag_profile]).argsort(axis = 0).flatten()[:-n:-1] , ids)
    
    return tag_array.index[ids].tolist()

    
def get_tag_based_inference(tag_profile, tag_array, title2handle = None , ids = None, standalone = False, n_recos = 15):
    
    """
    note that product ids must be in list format
    """    
    #get actual tag based similar products
    tag_res = get_tags_sim(tag_profile, tag_array, n = n_recos)
    
    if not standalone:
        if ids:
            ids = [title2handle[item] for item in ids]
            #get similar products from the selected product styles by the user on popup page
            res = []
            size_each = n_recos//len(ids) + 1
            for pid in ids:
#                 print(base_url + pid)
                tag_profile_temp = tag_array.loc[pid]
                tag_array_temp = tag_array.loc[tag_res]
                res.extend( get_tags_sim( tag_profile_temp, tag_array_temp ,  n = size_each) )
        else:
            print('No products selected by the popup part')
            res = []
            
        #return mixed results
        return list(set(res))
    else:
        #return tag based recommendations only
        return tag_res



def get_user_tag_profile(connection, email, indices, db_name = 'lea_clothing_backend', collection_name = 'processed_user_profiles'):
    db = connection[db_name] #change the name of database as per your wish
    collection = db[collection_name] #change the collection
    
    values = collection.find_one({'_id': email})
    if values:
        values = values['value']
        return pd.Series(values, index = indices)
    else:
        raise Exception


def store_user_mongo_unprocessed(connection, data, email, db_name = 'lea_clothing_backend', collection_name = 'user_profiles'):
    db = connection[db_name] #change the name of database as per your wish
    collection = db[collection_name] #change the collection
    
    d = dict()
    d['_id'] = email
    d.update(data)
#     d.pop('email')
    
    #checking if user already exists in mongodb
    data = collection.find_one({'_id': email})
    if data:
        print(f'\nUser already exists in User_profiles, overwriting...\n')
        collection.replace_one({'_id': email},d)
    else:
        collection.insert_one(d)
        
        
def store_user_mongo(connection, tag_profile, email, db_name = 'lea_clothing_backend', collection_name = 'processed_user_profiles'):
    db = connection[db_name] #change the name of database as per your wish
    collection = db[collection_name] #change the collection
    
    d = dict()
    d['_id'] = email
    d['value'] = tag_profile.tolist()
    
    #checking if user already exists in mongodb
    data = collection.find_one({'_id': email})
    if data:
        print(f'User already exists in processed_user_profiles, overwriting...\n')
        collection.replace_one({'_id': email},d)
    else:
        collection.insert_one(d)


def model_fn_2(products_filename):
    processed_filename, title2handle = create_product_tags_arr(products_filename)
    
    productsXtags = pd.read_csv(processed_filename, header = 0, index_col = 0)
    
    base_url = 'https://leaclothingco.com/products/'
    
    return productsXtags, title2handle, base_url



