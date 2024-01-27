import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from sklearn.metrics.pairwise import cosine_similarity


# @st.cache_data
def load_model(model):
    return tf.keras.models.load_model(f'models/{model}/model')

@st.cache_data
def load_matrix(df_dataset, total_user, total_product):
    datatsets = np.zeros((total_user, total_product))
    for i, row in df_dataset.iterrows():
        datatsets[row['customer_id'] - 1, row['product_id']] = 1
    return datatsets

df_interaction = pd.read_csv('datasets/fashion/customer_interactions.csv', sep=',')
df_product = pd.read_csv('datasets/fashion/product_details.csv', sep=';')
df_history = pd.read_csv('datasets/fashion/purchase_history.csv', sep=';')

df_dataset = df_history[['customer_id', 'product_id']]
total_user = len(df_dataset['customer_id'].unique())
total_product = len(df_dataset['product_id'].unique())
matrix = load_matrix(df_dataset, total_user, total_product)

st.title('Recommendation System')
st.divider()

prediction_methods = ['By Customer ID', 'Select Products Manually']
prediction_method = st.radio('Prediction Method', prediction_methods)
if prediction_method == prediction_methods[0]:
    customer_ids = df_interaction['customer_id'].values.tolist()
    customer_id = st.selectbox('Select customer ID', customer_ids, index=0)
    df_buy = pd.merge(
        left=df_history,
        right=df_product,
        on=['product_id'],
    )
    df_buy = df_buy[df_buy['customer_id'] == customer_id]
    
else:
    df_buy = df_product.copy()
    df_buy['buy'] = False

    df_buy = st.data_editor(
        df_buy,
        column_config={
            "buy": st.column_config.CheckboxColumn(
                "Buy",
                help="Select product to buy",
                default=False,
            )
        },
        hide_index=True,
    )
    df_buy = df_buy[df_buy['buy']].drop(columns=['buy'])


st.divider()

st.header('Previously Purchased')
st.dataframe(df_buy, hide_index=True, use_container_width=True)
products_on_chart = df_buy['product_id'].tolist()

df_product_pred = pd.DataFrame([], columns=df_product.columns)
predict_customer = st.button('Predict', type='primary')
if predict_customer:
    st.session_state['products_on_chart'] = products_on_chart
    x = np.zeros((1, total_product))
    x[0, products_on_chart] = 1

    # # Neural Nets
    # model_nn = load_model('NN')
    # y_nn = model_nn(x)

    # Autoencoder
    model_fauto = load_model('autoencoder')
    y_auto = model_fauto(x)

    # Cosine Similarity
    item_similarity = cosine_similarity(matrix.T)
    item_scores = x.dot(item_similarity)[0]

    # Result
    df_product_pred = df_product.copy()
    # df_product_pred['neural_net'] = list(y_nn.numpy()[0])
    df_product_pred['autoencoder'] = list(y_auto.numpy()[0])
    df_product_pred['cosine'] = item_scores
    df_product_pred = df_product_pred.reset_index(drop=True)
    st.session_state['df_product_pred'] = df_product_pred

    
st.divider()
st.header('Recommendation Products')

if products_on_chart != st.session_state.get('products_on_chart', []):
    st.warning('Change has been made, click "Predict" to view result.')

if prediction_method == 'By Customer ID':
    st.info(':orange[**Note:**] predicting current user is better to use :green[Cosine Similarity].')

col_model, col_top = st.columns([2, 5])
model = col_model.radio('Model (Sort score by)', ['Autoencoder', 'Cosine Similarity'], index=1 if prediction_method == 'By Customer ID' else 0) #, disabled=prediction_method == 'By Customer ID')
n = col_top.slider('Top Recommendation Product', 1, 50, 10)

df_product_pred = st.session_state.get("df_product_pred", None)
if df_product_pred is None:
    st.warning('Please click predict to view the result.')
    st.stop()

df_product_pred = df_product_pred[~df_product_pred['product_id'].isin(products_on_chart)]
if model == 'Autoencoder':
    df_product_pred = df_product_pred.sort_values(by='autoencoder', ascending=False)
# elif model == 'Neural Networks':
#     df_product_pred = df_product_pred.sort_values(by='neural_net', ascending=False)
else:
    df_product_pred = df_product_pred.sort_values(by='cosine', ascending=False)
df_product_pred = df_product_pred.reset_index(drop=True)

print()
st.dataframe(
    df_product_pred.iloc[:n], 
    column_config={
        "price": st.column_config.NumberColumn(format="%.3f"),
        "ratings": st.column_config.NumberColumn(format="%.3f"),
        "autoencoder": st.column_config.ProgressColumn(
            "autoencoder",
            help="autoencoder",
            format="%.3f",
            min_value=0, #float(df_product_pred['autoencoder'].min()),
            max_value=1, #float(df_product_pred['autoencoder'].max()),
        ),
        "neural_net": st.column_config.ProgressColumn(
            "neural_net",
            help="neural_net",
            format="%.3f",
            min_value=0, #float(df_product_pred['neural_net'].min()),
            max_value=1, #float(df_product_pred['neural_net'].max()),
        ),
        "cosine": st.column_config.ProgressColumn(
            "cosine",
            help="cosine",
            format="%.3f",
            min_value=float(df_product_pred['cosine'].min()),
            max_value=float(df_product_pred['cosine'].max()),
        ),
    },
    use_container_width=True
)
