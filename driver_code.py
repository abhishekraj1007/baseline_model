from helper_code import beautify_recos, filter_results, recommend_without_tags, recommend_with_tags, get_similar_cart_items
from helper_code import  create_profile, store_user, get_tag_based_inference, store_user_unprocessed, beautify_recos
from helper_code import model_fn, filter_results

import os
import pandas as pd
import json, copy
import pickle
from flask import Flask, request, jsonify
app = Flask(__name__)

from flask_cors import CORS
CORS(app)

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine
username= 'lea_clothing'
password = 'leaclothing'
hostname = 'lea-clothing-db.curvyi9vuuc9.ap-south-1.rds.amazonaws.com'
postgre_port = '5432'
db_name = 'lea_clothing_db'
# opening connection to postgre
engine = create_engine(f'postgresql://{username}:{password}@{hostname}:{postgre_port}/{db_name}', pool_size=100, max_overflow=-1)
base_url = 'https://leaclothingco.com/products/'

#train model for the first time
model_fn(engine=engine)

@app.route("/check-user", methods = ['GET'])
def get_old_recos():
    """
    returns old recommendations plus old payload that user filled as form
    to be used as autofill on frontend
    """
    email = request.args.get("email")
    table_name = 'tags_profile_unproc'
    ## checking if profile exists
    with engine.connect() as con:
        recos = con.execute(f"""select "recos" from "{table_name}" where "email" = '{email}'""").fetchone()
        unproc_data = con.execute(f"""select "unproc_data" from "{table_name}" where "email" = '{email}'""").fetchone()
    #all required fields are being returned already from the past stored results
    return jsonify( {'response': json.loads(recos[0]) if recos else dict() ,
     'form_data': json.loads(unproc_data[0]) if unproc_data else dict() } )


@app.route('/test', methods=['GET'])
def get_test():
    return "Success....."


@app.route('/cart',methods=['POST'])
def cart():
    """
    To recommend similar products in the cart
    """
    data = request.json

    title2handle = pickle.load(open('title2handle', 'rb'))
    product_handle = title2handle[data['product_title']]
    email = data['email']
    
    results = get_similar_cart_items(email, product_handle, engine, similarity_matrix='sim')
    beautified_results = beautify_recos(recos = results, engine=engine)
    return jsonify(beautified_results)

@app.route('/recommend',methods=['POST'])
def recommend():
    data = request.json

    title2handle = pickle.load(open('title2handle', 'rb'))
    product_handle = title2handle[data['product_title']]
    email = data['email']
    results = -1

    user = pd.read_sql_query(f"""select * from "tags_profile" where "email" = '{email}'""",con=engine).set_index('email', drop = True)
    if not user.empty:
        user = user.iloc[0]
        try:
            results,display_text = recommend_with_tags(user, engine, reco_count=8)
            print('User tag profile found')
        except:
            pass
    if results == -1:
        print('User tag profile not found')
        results = recommend_without_tags(email, product_handle , engine, reco_count = 8)
        # set display text as normal recommendation engine is used
        display_text = 'Products based on your browsing history'
        

    beautified_results = beautify_recos(recos = results, engine=engine)
    return jsonify( {'beautified_results':beautified_results,'display_text':display_text} )


@app.route('/personalize',methods=['POST'])
def personalize():
    data = request.json
    data = data['finalQuizData']

    #data to be returned as it is in future and avoids the apostrophe error with SQL databases
    data_to_store = json.loads(json.dumps(copy.deepcopy(data)).replace("'","''"))

    email = data['email']['value']
    
    tag_profile = create_profile(data, product_tags_filename='product_tags')
    
    #Storing processed user profile
    store_user(tag_profile, email, engine)

    #if ids, StandAlone = False (also check inside function)
    #Getting Ids
    if data.get('styles', []):
        ids = data['styles']['value']
    else:
        ids = []
    
    ## passing postgre engine object to get tag based inference using tag array
    tag_plus_style = get_tag_based_inference(tag_profile, 'productsXtags' , engine , title2handle = 'title2handle', ids = ids,
                                                                            standalone = False, n_recos = 15)

    #filtering results based on user given price range
    price_dict = data["spend categories"]["value"]
    filtered_recos = filter_results(tag_plus_style, prices = price_dict,engine = engine)

    #getting all required fields
    beautified_results = beautify_recos(filtered_recos, engine, payload=data, take_size = True)

    #Storing Unprocessed data for future in a different collection
    store_user_unprocessed(email, data = data_to_store, recos = beautified_results, engine = engine)

    return jsonify( beautified_results )
    

if __name__ == '__main__':
    try:
        print('Entered code: __main__')
        app.run(port= os.environ.get('AWS_PORT', 5000) )
    except KeyboardInterrupt:
        print(f'Server closed.')
    except Exception as e:
        print('\nCODE CRASHED once due to: {e}\n')
    
