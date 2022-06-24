from helper_code import beautify_recos, get_inference
from helper_code import  create_profile, store_user, get_tag_based_inference, store_user_unprocessed, beautify_recos
from helper_code import process_products, pre_process

import os
import json
from flask import Flask, request, jsonify
app = Flask(__name__)

from flask_cors import CORS
CORS(app)

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine
# saving csv to postgresql
engine = create_engine('postgresql://lea_clothing:leaclothing@lea-clothing-db.curvyi9vuuc9.ap-south-1.rds.amazonaws.com:5432/lea_clothing_db')
base_url = 'https://leaclothingco.com/products/'



@app.route("/check-user/<email>", methods = ['GET'])
def get_old_recos(email):
    table_name = 'tags_profile_unproc'
    ## checking if profile exists
    with engine.connect() as con:
        data = con.execute(f"""select "recos" from "{table_name}" where "email" = '{email}'""").fetchone()
    
    #all required fields are being returned already from the past stored results
    return jsonify(json.loads(data[0])) if data else jsonify({})


@app.route('/test', methods=['GET'])
def get_test():
    return "Success....."


@app.route('/recommend',methods=['POST'])
def recommend():
    data = request.json
    # print(f"Recommendations for the product: { base_url + title2handle[ data['product_title'] ] }")
    results = get_inference(data['email'], data['product_title'], engine, reco_count = 8)

    return jsonify( [{'Handle':item, 'URL':base_url + item} for item in results] )


@app.route('/personalize',methods=['POST'])
def personalize():

    data = request.json
    data = data['finalQuizData']
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
                                                                            standalone = False, n_recos = 12)
    

    #getting all required fields
    results = beautify_recos(tag_plus_style, data, engine)

    #Storing Unprocessed data for future in a different collection
    store_user_unprocessed(email, data, recos = results, engine = engine)

    return jsonify( results )


if __name__ == '__main__':
    try:
        #to check if main getting executed
        print(f'\n---Entered main---\n')
        process_products(engine, sim_desc_flag=True)
        pre_process(engine)
        app.run(port= os.environ.get('AWS_PORT', 5000) )
    except KeyboardInterrupt:
        print(f'Server closed.')
    except Exception as e:
        print('\nCODE CRASHED once due to: {e}\n')
    
