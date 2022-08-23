from helper_code import beautify_recos, filter_results, recommend_without_tags, recommend_with_tags, get_similar_cart_items
from helper_code import  create_profile, store_user, get_tag_based_inference, store_user_unprocessed, beautify_recos
from helper_code import model_fn, filter_results
from helper_code import cronjob
from helper_code import update_product

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
schema_name = 'recommendmodel'
# opening connection to postgre

engine = create_engine(f'postgresql://{username}:{password}@{hostname}:{postgre_port}/{db_name}', pool_size=100, max_overflow=-1)

base_url = 'https://leaclothingco.com/products/'

#train model for the first time
model_fn(engine=engine, sim_desc_flag=False, crontype=False)

from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

scheduler = BackgroundScheduler()
# Scheduler will run once every 12 hours once in a day at XX:00:00
trigger = CronTrigger(
        year="*", month="*", day="*", hour="1/12", minute="0", second="0"
    )
scheduler.add_job(func=lambda: cronjob(engine), trigger=trigger)
scheduler.start()


@app.route('/test', methods=['GET'])
def get_test():
    return "Success....."


@app.route('/product-update-check', methods=['POST'])
def product_update_check():
    try:
        json_dict = request.json
        if len(json_dict) == 1:
            update_product(json_dict,engine, delete = True)
        else:
            update_product(json_dict,engine,table_name='products')
            # Updating variants
            for i in range(len(json_dict['variants'])):
                update_product(json_dict['variants'][i],engine,table_name='product_variants')
        
        # model_fn(engine, sim_desc_flag=False, crontype=True)
        
        return jsonify({
                            "status" : 200,
                            "message" : 'Success',
                            "response" : 'Product data modified.'
                        })

    except Exception as e:
        return jsonify({
                        "status" : 500,
                        "message" : [repr(e),str(e)],
                        "response" : None
                        })


@app.route('/update-styles', methods=['POST'])
def update_stylesheet():
    table_name = 'stylesheet'
    try:
        json_dict = request.json
        df = pd.DataFrame(json_dict, columns=['attribute', 'value', 'imgUrl'])
        df.to_sql(table_name, engine, index = False, schema = schema_name, if_exists = 'replace')
        return jsonify({
                            "status" : 200,
                            "message" : 'Success',
                            "response" : 'Styles were updated.'
                        })
    except Exception as e:
        return jsonify({
                        "status" : 500,
                        "message" : [repr(e),str(e)],
                        "response" : None
                        })


@app.route('/get-styles', methods = ['GET'])
def get_stylesheet():
    table_name = 'stylesheet'
    styles = pd.read_sql_query(f"""select * from {schema_name}."{table_name}" """,con=engine).to_dict('record')
    return jsonify(styles)


@app.route("/check-user", methods = ['GET'])
def get_old_recos():
    """
    returns old recommendations plus old payload that user filled as form
    to be used as autofill on frontend
    """
    try:
        email = request.args["email"]
        table_name = 'tags_profile_unproc'

        active_products = []
        ## checking if profile exists
        with engine.connect() as con:
            recos = con.execute(f"""select "recos" from {schema_name}."{table_name}" where "email" = '{email}'""").fetchone()
            unproc_data = con.execute(f"""select "unproc_data" from {schema_name}."{table_name}" where "email" = '{email}'""").fetchone()
        
        if recos:
        #filtering outdated product to avoid displaying as cached results
            product_names = pickle.load(open('product_names','rb'))
            for item in json.loads(recos[0]):
                handle = item['Handle']
                if handle in product_names:
                    active_products.append(item)

        #all required fields are being returned already from the past stored results
        return jsonify({
                            "status" : 200,
                            "message" : 'Success',
                            "response" : {'recos': active_products if active_products else {} ,
        'form_data': json.loads(unproc_data[0]) if unproc_data else {}}
                        })
    except Exception as e:
        return jsonify({
                        "status" : 500,
                        "message" : [repr(e),str(e)],
                        "response" : None
                        })


@app.route('/cart',methods=['GET'])
def cart():
    """
    To recommend similar products in the cart
    """
    try:
        email = request.args.get("email",' ')
        product_title = request.args['product_title']
        
        title2handle = pickle.load(open('title2handle', 'rb'))
        product_handle = title2handle[product_title]
        
        results = get_similar_cart_items(email, product_handle, engine, similarity_matrix='sim')
        beautified_results = beautify_recos(recos = results, engine=engine)
        return jsonify( {
                        "status" : 200,
                        "message" : 'Success',
                        "response" : beautified_results } )
    except Exception as e:
        return jsonify( {
                        "status" : 500,
                        "message" : [repr(e),str(e)],
                        "response" : None} )


@app.route('/recommend',methods=['GET'])
def recommend():
    try:
        email = request.args.get("email",' ')
        product_title = request.args['product_title']
        title2handle = pickle.load(open('title2handle', 'rb'))
        product_handle = title2handle[product_title]

        #initialize results as -1 to avoid adding slow exception checking
        results = -1

        user = pd.read_sql_query(f"""select * from {schema_name}."tags_profile" where "email" = '{email}'""",con=engine).set_index('email', drop = True)
        if not user.empty:
            user = user.iloc[0]
            results,display_text = recommend_with_tags(user, engine, reco_count=12)

        if results == -1:
            print('User tag profile not found')
            results = recommend_without_tags(email, product_handle , engine, reco_count = 12)
            # set display text as normal recommendation engine is used
            display_text = 'Products based on your browsing history'
            beautified_results = beautify_recos(recos = results, engine=engine)
        else:
            print('User tag profile found')
            ## Getting payload from unproc_json data
            table_name = 'tags_profile_unproc'
            with engine.connect() as con:
                unproc_data = con.execute(f"""select "unproc_data" from {schema_name}."{table_name}" where "email" = '{email}'""").fetchone()
            payload = json.loads(unproc_data[0])
            beautified_results = beautify_recos(recos = results, engine=engine,payload = payload, take_size =True)
        
        return jsonify( {
                        "status" : 200,
                        "message" : 'Success',
                        "response" : {'beautified_results':beautified_results,'display_text':display_text} } )
    
    except Exception as e:
        return jsonify( {
                        "status" : 500,
                        "message" : [repr(e),str(e)],
                        "response" : None } )


@app.route('/personalize',methods=['POST'])
def personalize():
    try:
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
                                                                                standalone = False, n_recos = 30)

        #filtering results based on user given price range
        price_dict = data["spend categories"]["value"]
        filtered_recos = filter_results(tag_plus_style, prices = price_dict,engine = engine)

        #getting all required fields
        beautified_results = beautify_recos(filtered_recos, engine, payload=data, take_size = True)

        #Storing Unprocessed data for future in a different collection
        store_user_unprocessed(email, data = data_to_store, recos = beautified_results, engine = engine)

        return jsonify( {
                        "status" : 200,
                        "message" : 'Success',
                        "response" : beautified_results } )
    except Exception as e:
        return jsonify( {
                        "status" : 500,
                        "message" : [repr(e),str(e)],
                        "response" : None} )
    

if __name__ == '__main__':
    try:
        print('Entered: __main__')
        app.run(port= os.environ.get('AWS_PORT', 5000) )
    except Exception as e:
        print('\nCODE CRASHED once due to: {e}\n')
    
