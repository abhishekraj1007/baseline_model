from helper_code import pre_process, get_sim, get_inference, model_fn
from helper_code import create_product_tags_arr, create_profile, store_user_mongo, get_tags_sim, get_tag_based_inference, model_fn_2, store_user_mongo,store_user_mongo_unprocessed

import os
import pymongo
import flask
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)
#activating pymongo collection
connection = pymongo.MongoClient("mongodb+srv://my_jurisdiction:P9N18hMrpvSWJiEy@graphql-cluster.lsiwf1p.mongodb.net/?retryWrites=true&w=majority")


# Load model 1
sim, users, avg_item_ratings, title2handle, base_url, tag_array = model_fn(orders_filename = 'orders_export_1.csv', products_filename='products_export_1.csv', tags_filename = 'productsXtags.csv')

#Load model 2
productsXtags, title2handle, base_url = model_fn_2(products_filename = 'products_export_1.csv')


# get data from the html form and perform prediction

@app.route('/test', methods=['GET'])
def get_test():
    return "Success....."

@app.route('/recommend',methods=['POST'])
def recommend():
    
    data = request.json
    
    print(f"Recommendations for the product: { base_url + title2handle[ data['product_title'] ] }")
    res = get_inference(data['email'], data['product_title'], sim, users, avg_item_ratings, title2handle, tag_array, connection)
    
    return jsonify( [{'Handle':item, 'URL':base_url + item} for item in res] )


    
# get data from the html form and perform prediction
@app.route('/personalize',methods=['POST'])
def personalize():
    print('check......')

    data = request.json
    data = data['finalQuizData']
    email = data['email']['value']
    
    
    tag_profile = create_profile(data, productsXtags_arr = productsXtags)
    
    #Storing Unprocessed data for future in a different collection
    store_user_mongo_unprocessed(connection, data, email, db_name = 'lea_clothing_backend', collection_name = 'user_profiles')
    
    #Storing processed user profile
    store_user_mongo(connection, tag_profile, email, db_name = 'lea_clothing_backend', collection_name = 'processed_user_profiles')
    
    #Getting Ids
    if data.get('styles', []):
        ids = data['styles']['value']
    else:
        ids = []
    
    tag_plus_style = get_tag_based_inference(tag_profile, tag_array = productsXtags, title2handle = title2handle, ids = ids,
                                                                            standalone = False, n_recos = 15)
                        
    
    return jsonify( [{'Handle':item, 'URL':base_url + item} for item in tag_plus_style] )


if __name__ == '__main__':
    while True:
        try:
            app.run(port= os.environ.get('HEROKU_PORT', 5000) )
        except Exception as e:
            print('Code crashed once due to:\n{e}')
            continue
    
    
