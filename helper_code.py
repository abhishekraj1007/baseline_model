##Things to test: Test process_product_name for correct conversions, check if the style matching criteria is correct,
## Check for if correct results and order is being returned after sampling results from the subresults

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from random import choices
import json

import pickle
from sklearn.metrics.pairwise import cosine_similarity

# defining macros
type_mappings = {
'Corset Top':'Tops',
'Shirts & Tops':'Tops',
'Bralette':'Tops',
'Blazer':'Tops',
'Crop Top':'Tops',
'Jacket':'Tops',
'Corset':'Tops',
'Top':'Tops',
'Sweater':'Tops',

'Dresses':'Dresses',
'Dress':'Dresses',
'Gown':'Dresses',
'Coord Set':'Dresses',
'Jumpsuit':'Dresses',
'Jumpsuits & Rompers':'Dresses',
'Skirt':'Dresses',

'Pants':'Bottoms',
'Pant':'Bottoms',
'Shorts':'Bottoms',
'Skirts':'Bottoms',
'Jeans':'Bottoms',

'Sweatshirt':'Loungewear',
'Loungewear':'Loungewear',
'Cardigan':'Loungewear',
'Joggers':'Loungewear',

'Detachable Sleeves':'Accessories',
'Belt':'Accessories',
'Gift Cards':'Accessories'
}

def pre_process(engine):
    #Read Orders file
    table_name = 'orders'
    orders = pd.read_sql_query(f"""select * from "[{table_name}]" """,con=engine)

    print(f'Started processing {table_name} for creating users:')

    #changing column names for convenience
    cols = orders.columns
    cols = [''.join(item for item in entity.split()) for entity in cols]
    orders.columns = cols

    title2handle = pickle.load(open('title2handle','rb'))
    product_names = pickle.load(open('product_names','rb'))

    #processing product names(Title conflicts)
    orders.lineitemname = orders.lineitemname.apply(lambda x: x.split(' - ')[0].strip())

    # converting product Titles to their Handles
    orders.lineitemname = orders.lineitemname.map(title2handle)
    
    #removing outdated products
    orders = orders.loc[orders.lineitemname.isin(product_names)]

    print(f'Preparing sim_demographics (sim_demo)')
    demo_df = orders.dropna(subset=['shippingprovincename'])
    sim_demo = pd.DataFrame([],index = demo_df.shippingprovincename.unique().tolist(), columns=demo_df.lineitemname.unique().tolist())
    for state,group in demo_df.groupby(['shippingprovincename'])['email','lineitemname']:
        for product in group.lineitemname.unique():
            users = group.loc[group.lineitemname == product, 'email'].unique()
            sim_products = group.loc[(group.email.isin(users))&(group.lineitemname!=product),'lineitemname'].value_counts().index.to_list()[:10]
            # print(state, product, sim_products, sep='::',end='\n\n')
            if sim_products:
                sim_demo.loc[state,product] = ','.join(sim_products)
    
    #saving hardcoded sim_demo for faster demographic inference
    sim_demo.to_sql('sim_demo', engine, index = True, if_exists = 'replace' )

    # Selecting required columns
    req = ['name', 'email','lineitemname','financialstatus','fulfillmentstatus']
    orders = orders[req]

    #selecting carts with two or more products
    order_counts = orders.name.value_counts()
    counts_index = order_counts[order_counts > 1].index

    # taking 2 or more cart values
    orders = orders.loc[orders.name.isin(counts_index)]
    
    #scoring user interactions
    rate_dict = {'paid':5, 'pending':4, 'voided':3, 'refunded':1}
    orders['rating'] = orders.financialstatus.map(rate_dict)
    
    orders = orders[['email', 'lineitemname', 'rating']]
    orders.reset_index(drop=True, inplace=True)

    # #preparing for missing products in training data to avoid errors at inference
    # missing_ids = list(set(product_names) - set(orders.lineitemname.unique()))
    # print(missing_ids)
    # for pid in missing_ids:
    #     orders.loc[len(orders.index)] = [str(pid) +'@Dummy', pid, 5]
    
    # pivoting tables for training data
    orders = orders.pivot_table('rating',['email'],'lineitemname')
    orders = orders.reindex(columns=product_names)
    orders.fillna(0, inplace = True)

    #dumping avg_item_ratings
    pickle.dump( orders.mean(axis = 0), open('avg_item_ratings','wb'))
    print(f'Users processing done.\n')
    #Finished processing orders ^
    
    ids = list(orders.columns)
    idx_to_ids = dict(enumerate(ids))
    # ids_to_idx = dict([(y,x) for x,y in idx_to_ids.items()])

    print(f'initializing corr matrix...')
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
    sim.to_sql('sim', engine, index = True, if_exists = 'replace' )
    print('Training Corr matrix finished,\nSaving weights.\n')



def process_products(engine, sim_desc_flag = False):
    """
    Function to initialize things and will be used for retraining, other purpose it serves:
    1. To check if products data has changed, if yes creates new product and tags mappings stored in db.
    2. Stores products tags in order, in order to be used later.
    3. Same as (2) for Product names.
    4. Dumps a Title to Header names mapping file.
    5. initializes tags profile database with schema
    """

    table_name = 'products'
    products = pd.read_sql_query(f'select * from "[{table_name}]"',con=engine)
    print(f'Started processing {table_name}:')

    #create sim_desc for description based product similarity and stores in postgreDB
    if sim_desc_flag == True:
        from description_helper import create_sim_desc
        print(f'Started creating similar description mappings')
        create_sim_desc(products, engine)

    ## dropping duplicates
    products.drop_duplicates(subset='title', keep="first", inplace= True)
    products.dropna(subset=['title','tags'], inplace = True)
    products.reset_index(inplace=True, drop = True)
    title2handle = dict(zip(products.title, products.handle))

    products = products[['handle','tags']]
    products.tags = products.tags.apply(lambda x: x.split(', ') if len(x)>1 else [])
    products = products.explode('tags')

    #drop rows having nan values as tags after the .explode step from list->rows
    products.dropna(subset=['tags'], inplace = True)

    products.tags = products.tags.apply(lambda x: ''.join( item.lower() for item in x.replace('-','').replace("'",'').split() ))
    products['count'] = 1
    products = products.pivot_table('count', ['handle'],'tags')
    products.fillna(0, inplace = True)
    
    ## dropping products list, Tags list, title2handle dict and saving productsXtags into postgre db
    product_names = products.index.to_list()
    product_tags = products.columns.to_list()
    pickle.dump(product_names, open('product_names','wb'))
    pickle.dump(product_tags, open('product_tags','wb'))
    pickle.dump(title2handle, open('title2handle','wb'))

    products.to_sql(name ='productsXtags', con=engine, index = True, if_exists = 'replace' )

    print(f'Setting up schema for processed and unprocessed user profiles\n')
    
    #pending check when to replace/append/delete
    temp = ['dummy@dummy'] + [0] * len(product_tags)
    empty_tag_profile = pd.DataFrame([temp], columns=['email'] + product_tags)
    empty_tag_profile.to_sql('tags_profile', engine, index = False, if_exists = 'replace')
    
    with engine.connect() as con:
        con.execute("""ALTER TABLE "tags_profile" ADD PRIMARY KEY ("email")""")
    
    #initialize tags_profile_unproc postgre table
    temp = ['dummy@dummy', json.dumps({'a':1}) ,
         json.dumps( {'handle':'item', 'URL':'url', 'title':'title', 'Size':'size', 'IMGURL':'img_url', 'Price':'price'} ) ]
    empty_tag_profile = pd.DataFrame([temp], columns=['email', 'unproc_data', 'recos'])
    empty_tag_profile.to_sql('tags_profile_unproc', engine, index = False, if_exists = 'replace')

    with engine.connect() as con:
        con.execute("""ALTER TABLE "tags_profile_unproc" ADD PRIMARY KEY ("email")""")

    ## pending
    ## add a function to check change in products/tags (check by reading old file)

    
def get_inference(email, product_title, engine, reco_count = 10, avg_item_ratings = 'avg_item_ratings', 
                        title2handle = 'title2handle', tag_array = 'productsXtags', similarity_matrix='sim'):
    
    title2handle = pickle.load(open(title2handle, 'rb'))
    product_handle = title2handle[product_title]

    temp_user = get_user(email, engine)
    sim = pd.read_sql_query(f'select * from {similarity_matrix}',con=engine).set_index('index', drop=True)
    
    # 1. Based on orders history
    if not temp_user.empty:
        print('Found existing user,')

        #Assume customer likes/Bought this product
        temp_user.loc[product_handle] = 5.0
        
        #Normalizing the rating scale by subtracting average ratings by users
        avg_item_ratings = pickle.load( open('avg_item_ratings','rb'))
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
        part1 = [product for (product,score) in res[:5]]
    else:
        part1 = []
        print('Customer order history not found, Skipping CF results')


    # Demographics Based
    # 2. Explore customers data and show frequently bought products in the customer's province
    try:
        customer_table = 'customers'
        province = pd.read_sql_query(f"""select "province" from "[{customer_table}]" where "email" = '{email}' """,con=engine)
        if not province.empty:
            province = province.iloc[0,0]
            data = pd.read_sql_query(f"""select "{product_handle}" from sim_demo where "index" = '{province}' """, con=engine)
            if not data.empty:
                part2 = data.iloc[0,0].split(',')[:5]
                print(f'Demographics results included for user from state: {province}')
            else:
                part2 = []
        else:
            part2 = []
    except Exception as e:
        print(f'Exception at Demographics part: {e}')
        part2 = []
    

    # 3. show content based recos
    # print( sim.loc[product_handle,:].nlargest(reco_count+1).index[1:])
    part3 = sim.loc[:,product_handle].sort_values(ascending = False)[1:6].index.to_list()
    
    
    # 4. get_similar descriptions based products
    try:
        data = pd.read_sql_query(f"""select "sim_products" from "sim_desc" where "product" = '{product_handle}' """, con = engine)
        if not data.empty:
            part4 = data.iloc[0,0].split(',')[:5]
        else:
            part4 = []
    except Exception as e:
        print(f'similar description part could not be loaded due to error: {e}')
        part4 = []


    # 5. get_tag_based_personalized recommendations else show similar tag based products
    tag_array = pd.read_sql_query(f'SELECT * FROM "{tag_array}"',con=engine).set_index('handle', drop = True)
    #fetching user attributes from mongodb
    tag_profile = get_user_tag_profile(email, tag_array.columns, engine)

    if not tag_profile.empty:
        print(f'Found tag profile: {email}')
        part5 = get_tag_based_inference(tag_profile, 'productsXtags' , engine ,
                            title2handle = 'title2handle', standalone = True, n_recos = 5)
    else:
        print(f'User tag profile not found')
        tag_profile = tag_array.loc[tag_array.index == product_handle ].iloc[-1,:]
        part5 = get_tag_based_inference(tag_profile, 'productsXtags' , engine ,
                        title2handle = 'title2handle', standalone = True, n_recos = 6)[1:]

    #print output and verify results
    # print(f'\nPart1: {part1},\nPart2: {part2},\n Part3:{part3},\n Part4:{part4},\n Part5:{part5}\n')
    
    # sampling recommendations based on priority based function
    # The five positions for weights represent prob dist of selection from each arr
    if temp_user.empty:
        Model_weights = [1, 1.5, 1, 1]
        results = weighted_sample_without_replacement( arrs= [ part2, part3, part4, part5 ], weight_each_arr = Model_weights, k=reco_count)
        return results
    else:
        # sample results from both collaborative(60%) and rest(40%) from content+demographics+tags(personalized/Non-personalized)+similar_desc
        Model_weights = [10, 2, 3, 3, 5]
        results = weighted_sample_without_replacement( arrs= [ part1, part2, part3, part4, part5 ], weight_each_arr = Model_weights, k=reco_count)
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

######################################################################


def create_profile(data, product_tags_filename = 'product_tags'):
    """
    function to process raw json into useful user profile
    """
    product_tags = pickle.load(open(product_tags_filename, 'rb'))
    
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
                ''.join( item.lower() for item in tag.replace('-','').replace("'",'').split() ),weight) for tag in temp ))
            
#             print(tags)
        else:
            values = data.get(item, [])
            if values:
                values = values['value']
                
            for tag in values:
                value = uncomfortable_dict[tag]
                
                # VERIFY tags coming from payload in the end 
                tags.update(dict( (
                    ''.join( item.lower() for item in key.replace('-','').replace("'",'').split() ), value * weight) for key,value in value.items() ))
#             print('uncomfortable', values, tags)
    #using product_tags as columns/index of the tag profile
    tag_profile = pd.Series(tags, index = product_tags)
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

    
def get_tag_based_inference(tag_profile, tag_array, engine, title2handle = None , ids = None, standalone = False, n_recos = 15):
    """
    note that product ids must be in list format
    """
    tag_array = pd.read_sql_query(f'SELECT * FROM "{tag_array}"',con=engine).set_index('handle', drop = True)

    ## get actual tag based similar products
    tag_res = get_tags_sim(tag_profile, tag_array, n = n_recos)
    
    if not standalone:
        if ids:
            title2handle = pickle.load(open(title2handle, 'rb'))
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
            res = tag_res
            
        #return mixed results
        return list(set(res))
    else:
        #return tag based recommendations only
        return tag_res



def get_user_tag_profile(email, indices, engine):
    with engine.connect() as con:
        values = con.execute(f"""select * from "tags_profile" where "email" = '{email}'""").fetchone()
    
    if values:
        #skipping email and creating profile with the sparse values
        return pd.Series(values[1:], index = indices)
    else:
        return pd.Series([])


def store_user_unprocessed(email, data, recos, engine):
    """
    1. email
    2. data is out payload sent to be as it is
    3. recos are the recommendations from the personalize part of the API in json format
    to be stored
    """
    table_name = 'tags_profile_unproc'

    ## checking if profile exists
    with engine.connect() as con:
        user = con.execute(f"""select "email" from "{table_name}" where "email" = '{email}'""").fetchone()
    if user:
        print(f'Updating old Unprocessed data: {user[0]}')
        with engine.connect() as con:
            con.execute(f"""
            UPDATE "{table_name}"
            SET "email" = '{email}',
            "unproc_data" = '{json.dumps(data)}',
            "recos" = '{json.dumps(recos)}'
            WHERE "email" = '{email}'
            """)
    else:
        # Insert data as user profile not exists
        with engine.connect() as con:
            con.execute(f"""
            insert into "{table_name}"
            values ('{email}', '{json.dumps(data)}', '{json.dumps(recos)}')
            """)
        print(f'New user added in unprocessed Data.')


def store_user(tag_profile, email, engine):
    table_name = 'tags_profile'

    ## checking if profile exists
    with engine.connect() as con:
        data = con.execute(f"""select "email" from "{table_name}" where "email" = '{email}'""").fetchone()
    if data:
        print(f'Updating current User profile : {data[0]}\nand')
        s= ''
        for idx,value in zip(tag_profile.index, tag_profile.values):
            temp = idx + '=' + str(int(value)) + ','
            s+= temp
        s = s[:-1]
        with engine.connect() as con:
            con.execute(f"""
            UPDATE "{table_name}"
            SET {s}
            WHERE "email" = '{email}'
            """)
    else:
        # Insert data as user profile not exists
        with engine.connect() as con:
            con.execute(f"""
            insert into "{table_name}"
            values ('{email}', {str(tag_profile.values.tolist())[1:-1]} )
            """)
        print('New user tag profile added.')


def get_user(email, engine):
    """
    to get user order history at runtime(get_inference) to be used with .sim matrix -> doctorized product ratings for inference
    """
    table_name = 'orders'
    orders = pd.read_sql_query(f"""select * from "[{table_name}]" where "email" = '{email}'""",con=engine)
    cols = orders.columns
    cols = [''.join(item for item in entity.split()) for entity in cols]
    orders.columns = cols

    # Selecting required columns
    req = ['name', 'email','lineitemname','financialstatus','fulfillmentstatus']
    orders = orders[req]
    #processing product names(Handle- Title conflicts)
    orders.lineitemname = orders.lineitemname.apply(lambda x: x.split(' - ')[0].strip())
    title2handle = pickle.load(open('title2handle','rb'))
    # converting product Titles to their Handles
    orders.lineitemname = orders.lineitemname.map(title2handle)
    #loading handle names
    product_names = pickle.load(open('product_names','rb'))
    #removing outdated products
    orders = orders.loc[orders.lineitemname.isin(product_names)]
    if len(orders) == 0:
        return pd.Series([])
    
    # if data is not empty proceed further
    user_profile = pd.Series(index = product_names, dtype = 'int')
    rate_dict = {'paid':5, 'pending':4, 'voided':3, 'refunded':1}
    orders['rating'] = orders.financialstatus.map(rate_dict)
    user_profile[orders.lineitemname.values] = orders.rating.values
    
    # print(user_profile[user_profile!=0])
    return user_profile    


def beautify_recos(recos, engine, payload = None, take_size = False):
    """
    returns required fields in dict/json format with the product's handle(S) as input,
    dont forget to input list of products as input.
    eg:
    'Handle':item, 'URL':url, 'Title':title, 'Size':size, 'IMGURL':img_url, 'Price':price
    """
    table_name = 'products'
    base_url = 'https://leaclothingco.com/products/'

    if take_size:
        try:
            size = get_size(payload)
        except:
            size = 'NA'
    else:
        size = ''

    res = []
    for product in recos:
        handle = product
        try:
            # square brackets [ ] are used to refer to a SQL view
            title = pd.read_sql_query(f"""select "title" from "[{table_name}]" where "handle" = '{product}' """,con=engine).iloc[0,0]
        except Exception as e:
            print(f'Invalid product found after inference: CHECK! {e}')
            continue
        img_url = pd.read_sql_query(f"""select "img_url" from "[{table_name}]" where "handle" = '{product}' """,con=engine).iloc[0,0]
        values = pd.read_sql_query(f"""select "price" from "[{table_name}]" where "handle" = '{product}' """,con=engine).values.flatten()
        if len(set(values)) > 1:
            price = f'Starts from {int(min(values))}'
        else:
            price = f'{int(values[0])}'
        url = base_url + handle
        res.append( { 'Handle':handle, 
        'URL':url, 
        'Title':title, 
        'IMGURL':img_url, 
        'Price':price, 
        'Size':size } )
    return res


def get_size(payload):
    """
    returns a size L,M,XL,XXL based on body measurements from the provided payload data
    """
    size_chart = {"bust_size":{"32-33.5":"XS",
                    "34-35":"S",
                    "35.5-37":"M",
                    "37.5-39.5":"L",
                    "40-42":"XL",
                    "42.5-45.5":"XXL",
                    "46-48":"3XL",
                    "48.5-50.5":"4XL",
                    "51-53":"5XL"},
        
        "waist_size":{"23-24.5":"XS",
                    "25-26.5":"S",
                    "27-28.5":"M",
                    "29-32":"L",
                    "33.5-34.5":"XL",
                    "35-37":"XXL",
                    "37.5-39.5":"3XL",
                    "40-42":"4XL",
                    "42.5-44.5":"5XL"},

        "hip_size":{"34-35.5":"XS",
                    "36-37.5":"S",
                    "38-39.5":"M",
                    "40-42":"L",
                    "42.5-44.5":"XL",
                    "45-47.5":"XXL",
                    "48-50.5":"3XL",
                    "51-53":"4XL",
                    "53.5-55.5":"5XL"}
        }

    size_dict = {}
    size_list = [size_chart['bust_size'][payload["size"]['value']['Bust']] ,size_chart["waist_size"][payload["size"]['value']['Waist']] ,size_chart["hip_size"][payload["size"]['value']['Hips']]]
    for  size in size_list:
      if size not in size_dict:
        size_dict[size] = 1
      else:
        size_dict[size] = size_dict[size]+1

    size = max(size_dict, key= lambda x: size_dict[x])

    if size_dict[size]==1:
      return size_chart["waist_size"][payload["size"]['value']['Waist']] 
    else:
      return size
    

def filter_results(recos, prices, engine):
    """
    function to filter results based on price range inputs

    usage: filter_results(['coco-pink-polka-dot-corset-midi-dress','belle-black-ruffle-tulle-puff-corset-dress'], prices, engine)
    """
    print('Filtering results...')
    # create a dict-list of recos with prices
    table_name = 'products'

    data = []

    for product_handle in recos:
        values = pd.read_sql_query(f"""select "price" from "[{table_name}]" where "handle" = '{product_handle}' """,
                                con=engine).values.flatten()
        price = (int(min(values)), int(max(values)) )

        product_type = pd.read_sql_query(f"""select "product_type" from "[{table_name}]" where "handle" = '{product_handle}' """,
                                                                                            con=engine).iloc[0,0]
        custom_type = type_mappings.get(product_type, "Dresses")
        
        ## 0: product_handle, 1: product type, 2:prices(low/high)
        data.append( (product_handle, custom_type, price) )
    
    ## finally filter results based on type of clothe
    results = []
    for product in data:
        #select product's highest and lowest price
        product_low = product[2][0]
        product_high = product[2][1]
        # select product type
        custom_type = product[1]

        # find users lowest and highest budget based on dress type
        user_low = prices[custom_type][0]
        user_high = prices[custom_type][1]

        # check if product's lower is greater than user's and higher is lower than user's AND if product price range is very large
        if not ((product_high < user_low) or (product_low > user_high)):
            # append results
            results.append(product[0])

    print('filtered.')
    # return product_handles
    return results

def model_fn(engine):
    process_products(engine, sim_desc_flag=False)
    pre_process(engine)