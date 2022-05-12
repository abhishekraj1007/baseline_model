from helper_code import pre_process, get_sim, get_inference, model_fn
import pandas as pd
from flask import Flask, render_template , request
import pickle
import flask
app = Flask(__name__)

sim, users, avg_item_ratings, id2product, base_url = model_fn(orders_filename = 'orders_export_1.csv')


@app.route('/')
# @app.route('/index')
def index():
    return flask.render_template('index_page.html')


# get data from the html form and perform prediction
@app.route('/predict',methods=['POST'])
def predict():
    data = dict(request.form)
    if  int(data['product_id']) <= int(sim.index[-1]):
        print(f"Recommendations for the product: {base_url + id2product[ int( data['product_id'] ) ] }")
        res = get_inference(data['email'], int(data['product_id']), sim, users, avg_item_ratings, id2product)
    #     print(*res,sep='\n')
        return render_template("index_page.html", prediction_text=res)
    else:
        return render_template("index_page.html", prediction_text=[])


if __name__ == '__main__':
    while True:
        try:
            app.run(port=5000, debug=False)
        except Exception as e:
            print('Code crashed once due to:\n{e}')
            continue
    