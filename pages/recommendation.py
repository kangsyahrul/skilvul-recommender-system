import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_model():
    return tf.keras.models.load_model('models/FCN/model')

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

st.title('Prediction')
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
        # disabled=["widgets"],
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
    x = np.zeros((1, total_product))
    x[0, products_on_chart] = 1

    # Deep Learning
    model = load_model()
    y = model(x)

    # Cosine Similarity
    item_similarity = cosine_similarity(matrix.T)

    # Result
    df_product_pred = df_product.copy()
    df_product_pred['score_nn'] = list(y.numpy()[0])
    df_product_pred['score_cosine'] = x.dot(item_similarity)[0]
    df_product_pred = df_product_pred.reset_index(drop=True)
    st.session_state['df_product_pred'] = df_product_pred

    
st.divider()
st.header('Recommendation Products')

col_model, col_top = st.columns([2, 5])
model = col_model.radio('Model (Sort score by)', ['Neural Networks', 'Cosine Similarity'])
n = col_top.slider('Top Recommendation Product', 1, 50, 10)

df_product_pred = st.session_state.get("df_product_pred", pd.DataFrame([], columns=df_product.columns))
df_product_pred = df_product_pred[~df_product_pred['product_id'].isin(products_on_chart)]
df_product_pred = df_product_pred.sort_values(by='score_nn' if model == 'Neural Networks' else 'score_cosine', ascending=False)
df_product_pred = df_product_pred.reset_index(drop=True)

print()
st.dataframe(
    df_product_pred.iloc[:n], 
    column_config={
        "price": st.column_config.NumberColumn(format="%.3f"),
        "ratings": st.column_config.NumberColumn(format="%.3f"),
        "score_nn": st.column_config.ProgressColumn(
            "score_nn",
            help="score_nn",
            format="%.3f",
            min_value=float(df_product_pred['score_nn'].min()),
            max_value=float(df_product_pred['score_nn'].max()),
        ),
        "score_cosine": st.column_config.ProgressColumn(
            "score_cosine",
            help="score_cosine",
            format="%.3f",
            min_value=float(df_product_pred['score_cosine'].min()),
            max_value=float(df_product_pred['score_cosine'].max()),
        ),
    },
    use_container_width=True
)
