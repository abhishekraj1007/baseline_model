from helper_code import get_inference
from helper_code import  create_profile, store_user_mongo, get_tag_based_inference, store_user_mongo,store_user_mongo_unprocessed

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine
# import psycopg2
username = "postgres"
password = "vivek"

# saving csv to postgresql
engine = create_engine('postgresql://postgres:vivek@localhost:5432/test')
base_url = 'https://leaclothingco.com/products/'


app = Flask(__name__)

from flask_cors import CORS
CORS(app)


# get data from the html form and perform prediction

@app.route('/test', methods=['GET'])
def get_test():
    return "Success....."


@app.route('/recommend',methods=['POST'])
def recommend():

    data = request.json
    
    # print(f"Recommendations for the product: { base_url + title2handle[ data['product_title'] ] }")
    res = get_inference(data['email'], data['product_title'], engine, reco_count = 8)

    return jsonify( [{'Handle':item, 'URL':base_url + item} for item in res] )

    
# get data from the html form and perform prediction
@app.route('/personalize',methods=['POST'])
def personalize():


    data = request.json
    data = data['finalQuizData']
    email = data['email']['value']
    
    tag_profile = create_profile(data, product_tags_filename='product_tags')
    
    #Storing Unprocessed data for future in a different collection
    # store_user_mongo_unprocessed(connection, data, email, db_name = 'lea_clothing_backend', collection_name = 'user_profiles')
    
    #Storing processed user profile
    store_user_mongo(tag_profile, email, engine)

    #if ids, StandAlone = False (also check inside function)
    #Getting Ids
    if data.get('styles', []):
        ids = data['styles']['value']
    else:
        ids = []
    
    ## passing postgre engine object to get tag based inference using tag array
    tag_plus_style = get_tag_based_inference(tag_profile, 'productsXtags' , engine , title2handle = 'title2handle', ids = ids,
                                                                            standalone = False, n_recos = 12)
                      
    return jsonify( [{'Handle':item, 'URL':base_url + item} for item in tag_plus_style] )


if __name__ == '__main__':
    try:
        app.run(port= os.environ.get('HEROKU_PORT', 5000) )
    except KeyboardInterrupt:
        print(f'Server closed.')
    except Exception as e:
        print('\nCODE CRASHED once due to: {e}\n')
    
