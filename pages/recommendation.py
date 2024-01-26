import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

@st.cache_data
def load_model():
    return tf.keras.models.load_model('models/FCN/model')


df_interaction = pd.read_csv('datasets/fashion/customer_interactions.csv', sep=',')
df_product = pd.read_csv('datasets/fashion/product_details.csv', sep=';')
df_history = pd.read_csv('datasets/fashion/purchase_history.csv', sep=';')

total_product = len(df_product['product_id'].unique())

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
st.markdown(f'products_on_chart: {products_on_chart}')

df_product_pred = pd.DataFrame([], columns=df_product.columns)
predict_customer = st.button('Predict', type='primary')
if predict_customer:
    input_arr = np.zeros((1, total_product))
    input_arr[0, products_on_chart] = 1

    x_input = np.array(input_arr)
    model = load_model()
    pred = model(x_input)

    df_product_pred = df_product.copy()

    df_product_pred['Score'] = pred.numpy()[0]
    df_product_pred = df_product_pred[~df_product_pred['product_id'].isin(products_on_chart)]
    df_product_pred = df_product_pred.sort_values(by=['Score'], ascending=False).head(10)
    df_product_pred = df_product_pred.reset_index(drop=True)
    st.session_state['df_product_pred'] = df_product_pred
    
st.divider()
st.header('Recommendation Product')

n = st.slider('Top Recommendation Product', 1, 20, 10)
df_product_pred = st.session_state.get("df_product_pred", pd.DataFrame([], columns=df_product.columns))
st.dataframe(df_product_pred.iloc[:n], use_container_width=True)
