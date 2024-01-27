import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.title('Product Recommendation System')
'''Welcome to :orange[**Product Recommender System**]. This app utilize public dataset that can be 
found in [bhanupratapbiswas/fashion-products](https://www.kaggle.com/datasets/bhanupratapbiswas/fashion-products).

For more details about this code, you can see the [github code here](https://github.com/kangsyahrul/skilvul-recommender-system).

This app contains several pages that can be used by marketing team:
1. [analysis](/analysis): exploratory data analysis
2. [recommendation](/recommendation): predict what will be next item purchased
'''
st.divider()

st.header('Data Processing')
'''
Due to limitation of dataset, we use public dataset that is normalize using python notebook at `notebooks/datasets-fashion.ipynb`.
This notebook converts the data set into the format given in the test.

Additional filed such as `page_view` and `time_spent` are not exists so we will create based on given dataset statistic.
'''
st.divider()

st.header('Recommender System Algorithm')
'''
This apps uses several algorithms to recommend products to the customer. 
1. `Cosine Similiarity`: search other user that has similar personalization and recommends the products. :orange[It is good for predicting next items from current user].
3. `Autoencoder`: use autoencoder architecture shown in below image by mapping current purchased products with other products that are likely to purchase. :blue[It is good for predicting from custom items].
'''
st.image('images/autoencoder.png', 'Autoencoder architecture')

'''To use autoencoder, we need to build dataset by :blue[randomly masking actual purchased product] as `input` while `output` is the :violet[actual purchased product] by the customer.

For example:
'''
st.dataframe(
    pd.DataFrame({
        'category': ['Dress', 'Jeans', 'Shoes', 'Sweater', 'T-shirt'],
        'input (masked)': [0, 1, 0, 0, 0],
        'output (actual purchased product)': [1, 1, 0, 1, 0],
    })
)
'''The model was already trained using notebook at `notebooks/modelling.ipynb`'''
st.divider()

st.header('Web Application')
'''
This apps uses [streamlit.io](https://streamlit.io/) python based web frame work and deploy to [Azure Webb App](https://azure.microsoft.com/en-us/products/app-service/web).

How to run this app locally (code in macos/linux based, windows OS need to be adjusted):
1. Create virtual environment
    - `python -m venv venv`
2. Activate environment
    - `source venv/bin/activate`
3. Install all dependencies
    - `pip3 install -r requirements.txt`
4. Run web app
    - `streamlit run main.py`

Deploying app to Azure requires Microsoft Account and can not be explain here. 
Tutorial how to deploy to azure web app can be [find here](https://learn.microsoft.com/en-us/azure/app-service/quickstart-python?tabs=flask%2Cmac-linux%2Cazure-cli%2Cvscode-deploy%2Cdeploy-instructions-azportal%2Cterminal-bash%2Cdeploy-instructions-zip-azcli).
'''