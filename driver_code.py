from helper_code import pre_process, get_sim, get_inference
import pandas as pd
from flask import Flask, render_template , request
import pickle
import flask
app = Flask(__name__)
# base_url = 'https://leaclothingco.com/products/'



@app.route('/')
# @app.route('/index')
def index():
    return flask.render_template('index_page.html')


# get data from the html form and perform prediction
@app.route('/predict',methods=['POST'])
def predict():
    data = dict(request.form)
#     user_id = [x for x in request.form.values()]
    print(f"Recommendations for the product: {base_url + id2product[ int( data['product_id'] ) ] }")
    res = get_inference(data['email'], int(data['product_id']), sim, users, avg_item_ratings, id2product)
#     print(*res,sep='\n')
    return render_template("index_page.html", prediction_text=res)


def model_fn(orders_filename):
    users_filename, id2product = pre_process(orders_filename)
    
    get_sim(users_filename)
    sim = pd.read_csv('sim.csv', index_col = 0, header = 0)
    users = pd.read_csv('users.csv', index_col = 0, header = 0)
    
    avg_item_ratings = users.mean(axis = 0)
    
    base_url = 'https://leaclothingco.com/products/'
    
    return  sim, users, avg_item_ratings, id2product, base_url


if __name__ == '__main__':
    global sim, users, avg_item_ratings, id2product, base_url
    sim, users, avg_item_ratings, id2product, base_url = model_fn(orders_filename = 'orders_export_1.csv')
    app.run(port=5000, debug=False)
    