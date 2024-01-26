import pandas as pd
import streamlit as st


df_interaction = pd.read_csv('datasets/fashion/customer_interactions.csv', sep=',')
df_product = pd.read_csv('datasets/fashion/product_details.csv', sep=';')
df_history = pd.read_csv('datasets/fashion/purchase_history.csv', sep=';')

total_product = len(df_product['product_id'].unique())

st.title('Database')
tab_propduct, tab_customer, tab_history = st.tabs(['Product', 'Customers', 'Transaction'])
with tab_propduct:
    st.dataframe(df_product, hide_index=True,)

with tab_customer:
    st.dataframe(df_interaction, hide_index=True,)

with tab_history:
    st.dataframe(df_history, hide_index=True,)

st.divider()
