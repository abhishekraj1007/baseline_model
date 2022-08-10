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
from datetime import datetime

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
'Gift Cards':'Accessories',

'Saree':'Ethnicwear',
'Blouse':'Ethnicwear',
'Lehenga':'Ethnicwear',
'Gown':'Ethnicwear'
}


#{ (tag, abstract_value) : (message_to_be_displayed, [tags_to_be_used_for_recommendations], [actual_weightages]) }
tags_to_question = {
('hourglass','pos'):('Because you described your body as \n"Hourglass"',['hourglass'],[1]),
('pear','pos'):('Because you described your body as \n"pear"',['pear'],[1]),
('rectangle','pos'):('Because you described your body as \n"Rectangle"',['rectangle'],[1]),
('apple','pos'): ('Because you described your body as \n"Apple"',['apple'],[1]),
('invertedtriangle','pos'): ('Because you described your body as \n"Inverted Triangle"',['invertedtriangle'],[1]),

('sleeveless','pos'):('Because you like to highlight your \n"Arms"',['sleeveless'],[1]),
('corset','pos'):('Because you like to highlight your \n"Waist"',['corset','bodycon','croptop'],[1,1,1]),
('bodycon','pos'):('Because you like to highlight your \n"Waist"',['corset','bodycon','croptop'],[1,1,1]),
('croptop','pos'):('Because you like to highlight your \n"Waist"',['corset','bodycon','croptop'],[1,1,1]),
('mini','pos'):('Because you like to highlight your \n"Legs"',['mini'],[1]),
('backless','pos'):('Because you like to highlight your \n"Back"',['backless'],[1]),
('offshoulder','pos'):('Because you like to highlight your \n"Collarbones"',['offshoulder','strapless'],[1,1]),
('strapless','pos'):('Because you like to highlight your \n"Collarbones"',['offshoulder','strapless'],[1,1]),

('sleeves','pos'):('Because you like \n"Sleeves"',['sleeves'],1),
('bodycon','neg'):('Because you like \n"Bodycon"',['highwaist','skater','shift','slip','bodycon','croptop'],[1,1,1,1,-1,-1]),
('croptop','neg'):('Because you like \n"CropTop"',['highwaist','skater','shift','slip','bodycon','croptop'],[1,1,1,1,-1,-1]),
('highwaist','pos'):('Because you like \n"HighWaist"',['highwaist','skater','shift','slip','bodycon','croptop'],[1,1,1,1,-1,-1]),
('skater','pos'):('Because you like \n"Skater"',['highwaist','skater','shift','slip','bodycon','croptop'],[1,1,1,1,-1,-1]),
('shift','pos'):('Because you like \n"Shift"',['highwaist','skater','shift','slip','bodycon','croptop'],[1,1,1,1,-1,-1]),
('slip','pos'):('Because you like \n"Slip"',['highwaist','skater','shift','slip','bodycon','croptop'],[1,1,1,1,-1,-1]),
('midi','pos'):('Because you like \n"Midi"',['midi','gown','pants','maxi'],[1,1,1,1]),
('gown','post'):('Because you like \n"Gown"',['midi','gown','pants','maxi'],[1,1,1,1]),
('pants','post'):('Because you like \n"Pants"',['midi','gown','pants','maxi'],[1,1,1,1]),
('maxi','post'):('Because you like \n"Maxi"',['midi','gown','pants','maxi'],[1,1,1,1]),
('backless','neg'):('You may also like',['backless'],[-1]),
('strapless','neg'):('Because you like \n"Strapless"',['offshoulder','strapless'],[-1,-1]),
('offshoulder','neg'):('Because you like \n"off-shoulder"',['offshoulder','strapless'],[-1,-1]),

('petite','pos'):("Because you're Petite in height",['petite'],[1]),
('average','pos'):("Because you're Average in height",['average'],[1]),
('tall','pos'):("Because you're Tall in height",['tall'],[1]),

('pastel','pos'):("Because you like 'Pastels'",['pastel'],[1]),
('neutral','pos'):("Because you like 'Neutrals'",['neutral'],[1]),
('brighthues','pos'):("Because you like 'Bright Hues'",['brighthues'],[1]),
('earthytones','pos'):("Because you like 'Earthy Tones'",['earthytones'],[1]),
('neon','pos'):("Because you like 'Neons'",['neon'],[1]),

('floral','pos'): ("As you're fan of 'Prints'",['floral'],[1]),
('resort','pos'): ("As you're fan of 'Prints'",['resort'],[1]),
('3d','pos'): ("As you're fan of 'Prints'",['3d','handembroidery'],[1,1]),
('handembroidery','pos'): ("As you're fan of 'Prints'",['3d','handembroidery'],[1,1]),
('graphic','pos'): ("As you're fan of 'Prints'",['graphic'],[1]),
('geometric','pos'): ("As you're fan of 'Prints'",['geometric'],[1]),
('pearlembroidery','pos'): ("As you're fan of 'Prints'",['pearlembroidery'],[1]),
('sequins','pos'): ("As you're fan of 'Prints'",['sequins','rhinestones'],[1,1]),
('rhinestones','pos'): ("As you're fan of 'Prints'",['sequins','rhinestones'],[1,1]),

('birthday','pos'): ("As you're shopping for 'birthday'",['birthday'],[1]),
('graduation','pos'): ("As you're shopping for 'graduation'",['graduation'],[1]),
('mehendi','pos'): ("As you're shopping for 'Mehendi / Haldi'",['mehendi','haldi'],[1,1]),
('haldi','pos'): ("As you're shopping for 'Mehendi / Haldi'",['mehendi','haldi'],[1,1]),
('bridalshower','pos'): ("As you're shopping for 'Bridal Shower'",['bridalshower'],[1]),
('cocktail','pos'): ("As you're shopping for 'Cocktail' party",['cocktail'],[1]),
('datenight','pos'): ("As you're shopping for 'Date Night'",['datenight'],[1]),
('festivewear','pos'): ("As you're shopping for 'Festive Wear'",['festivewear'],[1]),
('beach','pos'): ("As you're shopping for 'Beach / Resortwear'",['beach','resortwear'],[1,1]),
('resortwear','pos'): ("As you're shopping for 'Beach / Resortwear'",['beach','resortwear'],[1,1]),
('weddingwear','pos'): ("As you're shopping for 'Wedding Wear'",['weddingwear'],[1]),
('party','pos'): ("As you're shopping for 'Party / Concert'",['party','concert'],[1,1]),
('concert','pos'): ("As you're shopping for 'Party / Concert'",['party','concert'],[1,1]),
('elevatedbasics','pos'): ("As you Like 'Elevated Basics'",['elevatedbasics'],[1])
}

schema_name = 'recommendmodel'


def process_products(engine, sim_desc_flag = True, crontype = False):
    """
    Function to initialize things and will be used for retraining, other purpose it serves:
    1. To check if products data has changed, if yes creates new product and tags mappings stored in db.
    2. Stores products tags in order, in order to be used later.
    3. Same as (2) for Product names.
    4. Dumps a Title to Header names mapping file.
    5. initializes tags profile database with schema
    """

    table_name = 'products'
    products = pd.read_sql_query(f'select * from {schema_name}."[{table_name}]"',con=engine)
    print(f'Started processing {table_name}')

    #create sim_desc for description based product similarity and stores in postgreDB
    if sim_desc_flag == True:
        from description_helper import create_sim_desc
        print(f'Started creating similar description mappings')
        try:
            create_sim_desc(products, engine)
        except Exception as e:
            print(f"Exception as similar description creation : {e}")

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
    pickle.dump(title2handle, open('title2handle','wb'))
    products.to_sql(name ='productsXtags', con=engine, index = True, schema = schema_name, if_exists = 'replace' )
    

    #check if table does not exists, then create new
    with engine.connect() as con:
        table_proc = con.execute(f"""SELECT to_regclass('{schema_name}.tags_profile')""").fetchone()
    if not table_proc[0]:
        print(f'Setting up schema for processed user profiles\n')
        #pending check when to replace/append/delete
        temp = ['dummy@dummy'] + [0] * len(product_tags)
        empty_tag_profile = pd.DataFrame([temp], columns=['email'] + product_tags)
        empty_tag_profile.to_sql('tags_profile', engine, index = False, schema = schema_name, if_exists = 'replace')
        
        with engine.connect() as con:
            con.execute(f"""ALTER TABLE {schema_name}."tags_profile" ADD PRIMARY KEY ("email")""")
    else:
        print('tags_profile table already exists.')

    #The purpose of putting this part of tag below tags_profile creation is to avoid those circumstances
    # where server needs to be run due to any issues
    # Also cronjob will not run without the above table, because it needs to compare with that.

    if crontype:
        print('Entered cron\nChecking if tags got changed...')
        #this file should exist when code runs this way through cron job
        try:
            old_tags = pickle.load(open('product_tags','rb'))
            pickle.dump(product_tags, open('product_tags','wb'))
            #check if tags got changed and we need to restructure the tag_array
            change_in_tags = set.symmetric_difference( set(product_tags), set(old_tags))
            if len(change_in_tags) > 0:
                update_tag_schema(engine)
            else:
                print('No tags were updated at croncheck.')
        except FileNotFoundError:
            print('Cron ran before Flask server or Couldnt found Old product tags file\n')
        except Exception as e:
            print(e,'  Error in Updating tag schema')
    
    #check if table does not exists, then create new
    with engine.connect() as con:
        table_unproc = con.execute(f"""SELECT to_regclass('{schema_name}.tags_profile_unproc')""").fetchone()
    if not table_unproc[0]:
        print(f'Setting up schema for unprocessed user profiles')
        #initialize tags_profile_unproc postgre table
        temp = ['dummy@dummy', json.dumps({'a':1}) ,
            json.dumps( {'handle':'item', 'URL':'url', 'title':'title', 'Size':'size', 'IMGURL':'img_url', 'Price':'price'} ) ]
        empty_tag_profile = pd.DataFrame([temp], columns=['email', 'unproc_data', 'recos'])
        empty_tag_profile.to_sql('tags_profile_unproc', engine, index = False, schema = schema_name, if_exists = 'replace')

        with engine.connect() as con:
            con.execute(f"""ALTER TABLE {schema_name}."tags_profile_unproc" ADD PRIMARY KEY ("email")""")
    else:
        print('tags_profile_Unproc table already exists.')

    #now save this file to let the model run normally
    pickle.dump(product_tags, open('product_tags','wb'))
        

def pre_process(engine):
    #Read Orders file
    table_name = 'orders'
    orders = pd.read_sql_query(f"""select * from {schema_name}."[{table_name}]" """,con=engine)

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
    
    #preparing ga_top_selling events (dumps a pickle file)
    print('Processing Google analytics top selling products')
    ga_top_selling(engine)
    
    #saving hardcoded sim_demo for faster demographic inference
    sim_demo.to_sql('sim_demo', engine, index = True, schema = schema_name, if_exists = 'replace' )

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
    sim.to_sql('sim', engine, index = True, schema = schema_name, if_exists = 'replace' )
    print('Training Corr matrix finished,\nSaving weights.\n')


def update_tag_schema(engine):
    """
    Update complete tags_profile table in postgre in case Tags metadata of products gets changed
    """ 
    tag_array = 'tags_profile'
    product_tags = pickle.load(open('product_tags','rb'))
    tag_profile = pd.read_sql_query(f'SELECT * FROM {schema_name}."{tag_array}"',con=engine).set_index('email', drop = True)

    if tag_profile.empty:
        print('Something went wrong, Tags profile is empty')
        return

    difference_tags = list(set(tag_profile.columns) - set(product_tags))
    print('These Tags were outdated: ',difference_tags)
    tag_profile.drop(difference_tags, axis = 1, inplace = True)

    res = []
    for i,row in enumerate(tag_profile.iterrows()):
        values = dict(zip(row[1].index, row[1].values))
        values['email'] = row[0]
        res.append(values)

    df = pd.DataFrame(res,columns=['email']+ product_tags)
    df.iloc[:,1:] = df.iloc[:,1:].fillna(0).astype(int)

    print(f'Setting up Updated Schema as Tags data got changed\n')

    df.to_sql('tags_profile', engine, index = False, schema = schema_name, if_exists = 'replace')

    with engine.connect() as con:
        con.execute(f"""ALTER TABLE {schema_name}."tags_profile" ADD PRIMARY KEY ("email")""")


def recommend_with_tags(user, engine, reco_count):
    """
    recommend based on random selected tags filled by the user for the recommend part on product page
    """
    # select filled user tags from db
    non_zero_attr = user.loc[user!=0]
    if not non_zero_attr.empty:
        random_tag = random.choice(non_zero_attr.index.to_list())
    #return empty if no tag found in user profile
    else:
        print('User tag profile found but no preferences filled by user,')
        return -1, ''
    
    #create key to be accessed from custom tags_to_question mappings
    key = (random_tag,'pos' if non_zero_attr[random_tag] > 0 else 'neg')
    # to be displayed on backend
    display_text = tags_to_question[key][0]
    #selecting 2nd and third position for tags and their weights for profile creation for custom recommendation part
    tags = dict(zip(tags_to_question[key][1], tags_to_question[key][2]))

    #get inference
    tag_profile = pd.Series(tags, index = user.index)
    tag_profile.fillna(0, inplace= True)
    result = get_tag_based_inference(tag_profile, 'productsXtags', engine, standalone = True, n_recos = reco_count)
    return result, display_text


def recommend_without_tags(email, product_handle, engine, reco_count = 10, avg_item_ratings = 'avg_item_ratings', 
                         tag_array = 'productsXtags', similarity_matrix='sim'):
    """
    will be used when user tags are not found in the db
    else recommend_with_tags will be used
    """
    
    temp_user = get_user(email, engine)
    sim = pd.read_sql_query(f'select * from {similarity_matrix}',con=engine).set_index('index', drop=True)
    
    # 1. Based on orders history (Collaborative filtering)
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


    # 2. show content based recos
    # print( sim.loc[product_handle,:].nlargest(reco_count+1).index[1:])
    part2 = sim.loc[:,product_handle].sort_values(ascending = False)[1:6].index.to_list()

    # Demographics Based
    # 3. Explore customers data and show frequently bought products in the customer's province
    try:
        customer_table = 'customers'
        province = pd.read_sql_query(f"""select "province" from {schema_name}."[{customer_table}]" where "email" = '{email}' """,con=engine)
        if not province.empty:
            province = province.iloc[0,0]
            data = pd.read_sql_query(f"""select {schema_name}."{product_handle}" from sim_demo where "index" = '{province}' """, con=engine)
            if not data.empty:
                part3 = data.iloc[0,0].split(',')[:5]
                print(f'Demographics results included for user from state: {province}')
            else:
                # print('Demographics data not found')
                part3 = []
        else:
            # print('Demographics data not found')
            part3 = []
    except Exception as e:
        print(f'Exception at Demographics part: {e}')
        part3 = []
    
    
    # 4. get_similar descriptions based products
    try:
        data = pd.read_sql_query(f"""select "sim_products" from {schema_name}."sim_desc" where "product" = '{product_handle}' """, con = engine)
        if not data.empty:
            part4 = data.iloc[0,0].split(',')[:5]
        else:
            part4 = []
    except Exception as e:
        print(f'similar description part could not be loaded due to error: {e}')
        part4 = []


    # 5. show similar tag based products
    tag_array = pd.read_sql_query(f'SELECT * FROM {schema_name}."{tag_array}"',con=engine).set_index('handle', drop = True)
    tag_profile = tag_array.loc[tag_array.index == product_handle ].iloc[-1,:]
    part5 = get_tag_based_inference(tag_profile, 'productsXtags' , engine , standalone = True, n_recos = 6)[1:]

    # 6. Top selling products info from Google Analytics
    part6 = pickle.load(open(f'ga_top_selling','rb'))[:5]

    #print output and verify results
    # print(f'\nPart1: {part1},\nPart2: {part2},\n Part3:{part3},\n Part4:{part4},\n Part5:{part5},\n Part6:{part6}')
    
    # sampling recommendations based on priority based function
    # The six positions for weights represent prob dist of selection from each arr
    if not temp_user.empty:
        # sample results from both collaborative(60%) and rest(40%) from content+demographics+tags(personalized/Non-personalized)+similar_desc
        Model_weights = [6, 2, 2, 2, 4, 1]
        results = weighted_sample_without_replacement( arrs= [ part1, part2, part3, part4, part5, part6 ], weight_each_arr = Model_weights, k=reco_count)
        return set(results)
    else:
        Model_weights = [1.5, 1, 1, 1, 1]
        results = weighted_sample_without_replacement( arrs= [ part2, part3, part4, part5, part6 ], weight_each_arr = Model_weights, k=reco_count)
        return set(results)
        

def ga_process_prod_name(prod_name):
    prod_name_unprocess = prod_name.split('/')[2]
    if '?' in prod_name_unprocess:
        prod_name_processed = prod_name_unprocess.split('?')[0]
    else:
        prod_name_processed = prod_name_unprocess
    return prod_name_processed


def ga_top_selling(engine):
    table_name = 'ga_events'
    df = pd.read_sql_query(f'select * from {schema_name}."[{table_name}]"',con=engine)

    df['path'] = df.path.apply(ga_process_prod_name)

    # filtering products which are expired or wrong names/coflicts
    products = pickle.load(open('product_names', 'rb'))
    df = df[ df.path.isin(products) ]

    df = df.loc[df.event_type == 'PRODUCT_VIEW']
    df = df.groupby(['path']).sessions.sum().reset_index().set_index('sessions',drop=True).sort_index(ascending = False)
    ga_top_selling = df.path.values.tolist()
    pickle.dump(ga_top_selling, open('ga_top_selling','wb'))
    print('Succesfully saved ga_top_Selling.')

        
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


def get_similar_cart_items(email, product_handle, engine, similarity_matrix='sim'):
    """"
    Recommendation APi for cart items
    """
    temp_user = get_user(email, engine)
    sim = pd.read_sql_query(f'select * from {schema_name}.{similarity_matrix}',con=engine).set_index('index', drop=True)
    
    # orders history
    if not temp_user.empty:
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
        res = [product for (product,score) in res[:5]]
    else:
        print('order history not found, using content based(sim) filtering for CART...')
        # print( sim.loc[product_handle,:].nlargest(reco_count+1).index[1:])
        res = sim.loc[:,product_handle].sort_values(ascending = False)[1:6].index.to_list()
    
    return res

######################################################################


def create_profile(data, product_tags_filename = 'product_tags'):
    """
    function to process raw json into useful user profile
    """
    product_tags = pickle.load(open(product_tags_filename, 'rb'))
    
    # a slightly smaller weight is assigned to filter tags
    uncomfortable_dict = {'Arms':{'Sleeves':1}, 
                        'Waist':{'High Waist':1, 'Skater':1, 'Shift':1, 'Slip':1, "Bodycon":-2, "Crop Top":-2},
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
    tag_array = pd.read_sql_query(f'SELECT * FROM {schema_name}."{tag_array}"',con=engine).set_index('handle', drop = True)

    ## get actual tag based similar products
    tag_res = get_tags_sim(tag_profile, tag_array, n = n_recos)
    
    if not standalone:
        if ids:
            title2handle = pickle.load(open(title2handle, 'rb'))
            #checking for invalid styles selection from hardcoded data on frontend
            ids = list( map( title2handle.get,filter(lambda x: True if x in title2handle.keys() else False, ids) ) )
            #get similar products from the selected product styles by the user on popup page
            res = []
            size_each = n_recos//len(ids) + 1
            for pid in ids:
#                 print(base_url + pid)
                tag_profile_temp = tag_array.loc[pid]
                tag_array_temp = tag_array.loc[tag_res]
                res.extend( get_tags_sim( tag_profile_temp, tag_array_temp ,  n = size_each) )
            #return mixed results
            return list(set(res))
        else:
            print('No Styles selected by the user, skipping..')
    #return tag based recommendations only
    return tag_res


def get_user_tag_profile(email, indices, engine):
    with engine.connect() as con:
        values = con.execute(f"""select * from {schema_name}."tags_profile" where "email" = '{email}'""").fetchone()
    
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
        user = con.execute(f"""select "email" from {schema_name}."{table_name}" where "email" = '{email}'""").fetchone()
    if user:
        print(f'Updating old Unprocessed data: {user[0]}')
        with engine.connect() as con:
            con.execute(f"""
            UPDATE {schema_name}."{table_name}"
            SET "email" = '{email}',
            "unproc_data" = '{json.dumps(data)}',
            "recos" = '{json.dumps(recos)}'
            WHERE "email" = '{email}'
            """)
    else:
        # Insert data as user profile not exists
        with engine.connect() as con:
            con.execute(f"""
            insert into {schema_name}."{table_name}"
            values ('{email}', '{json.dumps(data)}', '{json.dumps(recos)}')
            """)
        print(f'New user added in unprocessed Data.')


def store_user(tag_profile, email, engine):
    table_name = 'tags_profile'

    ## checking if profile exists
    with engine.connect() as con:
        data = con.execute(f"""select "email" from {schema_name}."{table_name}" where "email" = '{email}'""").fetchone()
    if data:
        print(f'Updating current User profile : {data[0]}')
        s= ''
        for idx,value in zip(tag_profile.index, tag_profile.values):
            temp = idx + '=' + str(int(value)) + ','
            s+= temp
        s = s[:-1]
        with engine.connect() as con:
            con.execute(f"""
            UPDATE {schema_name}."{table_name}"
            SET {s}
            WHERE "email" = '{email}'
            """)
    else:
        # Insert data as user profile not exists
        with engine.connect() as con:
            con.execute(f"""
            insert into {schema_name}."{table_name}"
            values ('{email}', {str(tag_profile.values.tolist())[1:-1]} )
            """)
        print('New user tag profile added.')


def get_user(email, engine):
    """
    to get user order history at runtime(get_inference) to be used with .sim matrix -> doctorized product ratings for inference
    """
    table_name = 'orders'
    orders = pd.read_sql_query(f"""select * from {schema_name}."[{table_name}]" where "email" = '{email}'""",con=engine)
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
    products = pd.read_sql_query(f"""select * from {schema_name}."[{table_name}]" """,con=engine)
    for product in recos:
        handle = product
        try:
            # square brackets [ ] are used to refer to a SQL view
            title = products.loc[products.handle == handle,'title'].iloc[0]
            img_url = products.loc[products.handle == handle,'img_url'].iloc[0]
            values = products.loc[products.handle == handle,'price'].values.flatten()
        except Exception as e:
            print(f'Invalid product found after inference: CHECK! {e}')
            continue
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
    print('Filtering results...',end='')
    # create a dict-list of recos with prices
    table_name = 'products'

    data = []

    for product_handle in recos:
        try:
            values = pd.read_sql_query(f"""select "price" from {schema_name}."[{table_name}]" where "handle" = '{product_handle}' """,
                                    con=engine).values.flatten()
            price = (int(min(values)), int(max(values)) )

            product_type = pd.read_sql_query(f"""select "product_type" from {schema_name}."[{table_name}]" where "handle" = '{product_handle}' """,
                                                                                                con=engine).iloc[0,0]
            custom_type = type_mappings.get(product_type, "Dresses")
            
            ## 0: product_handle, 1: product type, 2:prices(low/high)
            data.append( (product_handle, custom_type, price) )
        except:
            print('Invalid product found after inference')
    
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


def model_fn(engine, sim_desc_flag=True, crontype=False):
    process_products(engine, sim_desc_flag=sim_desc_flag, crontype = crontype)
    pre_process(engine)


def cronjob(engine):
    #initiate a log
    with open('lea_cron_log','a') as f:
        print(f'scheduler running at {datetime.now()}')
        f.write(str(datetime.now()) + '\n')
    #run model with crontype=True
    try:
        model_fn(engine, sim_desc_flag=True, crontype=True)
    except Exception as e:
        pass