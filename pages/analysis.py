import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

@st.cache_data
def load_graph_content(path):
    return open(path, 'r', encoding='utf-8').read() 

st.title('Exploratory Data Analysis')
st.divider()


# Dataset
st.header('Dataset')
df_interaction = pd.read_csv('datasets/fashion/customer_interactions.csv', sep=',')
df_product = pd.read_csv('datasets/fashion/product_details.csv', sep=';')
df_purchase = pd.read_csv('datasets/fashion/purchase_history.csv', sep=';')
df_customer = pd.merge(
        left=df_interaction,
        right=pd.merge(
            left=df_purchase,
            right=df_product,
            on=['product_id'],
        ).groupby('customer_id').agg({'product_id': 'count', 'price': 'sum'}).reset_index().rename(columns={'product_id': 'purchased_items'}),
        on=['customer_id'],
    )

df_dataset = df_purchase[['customer_id', 'product_id']]
total_user = len(df_dataset['customer_id'].unique())
total_product = len(df_dataset['product_id'].unique())

tab_propduct, tab_interaction, tab_purchased = st.tabs(['Product', 'Interactions', 'Purchased'])
with tab_propduct:
    st.dataframe(
        df_product, 
        column_config={
            "price": st.column_config.NumberColumn(format="%.3f"),
            "ratings": st.column_config.NumberColumn(format="%.3f"),
        },
        hide_index=True,
    )

with tab_interaction:
    st.dataframe(
        df_interaction, 
        hide_index=True,
    )

with tab_purchased:
    st.dataframe(
        df_purchase, 
        hide_index=True,
    )

st.divider()

# Customer Interactions
st.header('Customer Performance')

tab_customer, tab_items, tab_interaction = st.tabs(['Performance', 'Purchased Items', 'Interactions'])
with tab_customer:
    st.markdown('Show how many time and money they spend.')
    st.dataframe(df_customer)

with tab_items:
    st.markdown('How is the behaviour of the customer spend their time? How they spend their time and money?')
    fig = px.scatter(
        df_customer, 
        x="purchased_items", 
        y="price", 
        # color="price",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('There is no customer who buy a few items with high amount of money. This probably due to each product catalog has similar purpose and pricing are competitve.')


with tab_interaction:
    st.markdown('How is the behaviour of the customer spend their time? How they spend their time and money?')
    fig = px.scatter(
        df_customer, 
        x="page_views", 
        y="time_spent", 
        size="purchased_items",
        color="price",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('Mostly they who spend many time are likely to buy some products.')

st.divider()

# Product Performance
st.header('Product Performance')
'''Understanding how each product category perfromed in the market'''
tab_rating, tab_price, tab_purchased = st.tabs(['Ratings', 'Price', 'Total Purchased'])
with tab_rating:
    fig = px.box(
        df_product,
        x='category',
        y='ratings',
        title='Product Ratings',
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('Sweater has higher median ratings and probably most liked item.')

with tab_price:
    fig = px.box(
        df_product,
        x='category',
        y='price',
        title='Product Prices',
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('Dresses are relatively cheap.')

with tab_purchased:
    df_product_purchase = pd.merge(
        left=df_purchase,
        right=df_product,
        on=['product_id'],
    )
    df_product_performance = df_product_purchase.groupby(
        by='category'
    ).agg(
        {'customer_id': 'count', 'ratings': 'mean', 'price': 'mean'}
    ).reset_index().rename(
        columns={
            'customer_id': 'purchased',
            'ratings': 'ratings_avg',
            'price': 'price_avg',
    }
    ).melt(
        id_vars=["category"], 
        var_name="metrics", 
        value_name="value",
    ).sort_values(by=['category'])

    fig = px.bar(
        df_product_performance,
        x='category',
        y='value',
        color='metrics',
        title='Product Metrics',
        barmode='group',
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('Jeans mostly purchased products. ')
st.divider()


st.header('Price v.s Ratings')
'''Is there a relationship between price and rating? For example: do expensive products have good quality?'''
fig = px.scatter(
    df_product,
    x='price',
    y='ratings',
    color='category',
    title='Price v.s Ratings',
)
st.plotly_chart(fig, use_container_width=True)
st.markdown('Expensive products also have some small ratings. Products which have moderate price tend to have higher ratings.')
st.divider()

st.header('Product Relationship')
tab_cat, tab_sub = st.tabs(['Category', 'Brand'])
with tab_cat:
    grpah_content = load_graph_content('models/graph/category.html') 
    components.html(grpah_content, height=800)

with tab_sub:
    grpah_content = load_graph_content('models/graph/brand.html')
    components.html(grpah_content, height=800)
