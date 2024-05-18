import streamlit as st
from streamlit_option_menu import option_menu
from joblib import load
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(layout="wide")

st.title("Industrical coper modeling")
selected = option_menu ( 
                        
                        menu_title= None,
                        options=["Classification","Regression"],
                        icons=["house","house"],
                        orientation="horizontal"
                        
                        )
if selected == "Classification":
    with st.sidebar:
        df_features = pd.read_csv ("ml_features.csv")
        country = st.selectbox("country list",df_features['country'].unique())
        application = st.number_input("application")
        width = st.number_input("width")
        thickness = st.number_input("thickness")
        quantity = st.number_input("quantity tons")
        item_type_IPL= st.selectbox("item type_IPL",[0,1])
        item_type_Others= st.selectbox("item type_Others",[0,1])
        item_type_PL= st.selectbox("item type_PL",[0,1])
        item_type_S= st.selectbox("item type_S",[0,1])
        item_type_SLAWR= st.selectbox("item type_SLAWR",[0,1])
        item_type_W= st.selectbox("item type_W",[0,1])
        item_type_WI= st.selectbox("item type_WI",[0,1])
        
        item_date = st.date_input("item date",value=None)
        delivery_date = st.date_input("Delivery date",value=None)
        selling_price = st.number_input("Selling Price")
        
    classification = { 
                        "country":  country,
                        "application": application,
                        "width": width,
                        "thickness": thickness,
                        "quantity tons": quantity,
                        "item type_IPL": item_type_IPL,
                        "item type_Others": item_type_Others,
                        "item type_PL":item_type_PL,
                        "item type_S":item_type_S,
                        "item type_SLAWR":item_type_SLAWR,
                        "item type_W": item_type_W,
                        "item type_WI":item_type_WI,
                        "item_date": item_date,
                        "delivery date": delivery_date,
                        "selling_price": selling_price                   
                        }
        
    df_class = pd.DataFrame([classification]) 
    
    col1,col2 = st.columns(2)
    
    col1.markdown("## Input values")
    col1.table(df_class.T)

    df_class["item_date"] = pd.to_datetime(df_class["item_date"],format='%Y%m%d')
    df_class["delivery date"] = pd.to_datetime(df_class["delivery date"],format='%Y%m%d')
    
    df_class["quantity_kg"] = df_class["quantity tons"]*1000
    df_class["delivery_days"] = (df_class["delivery date"] - df_class["item_date"]).dt.days
    df_class['item_date_day'] = df_class['item_date'].dt.day
    df_class['item_date_month'] = df_class['item_date'].dt.month
    df_class['item_date_year'] = df_class['item_date'].dt.year
    df_class['delivery date_day'] = df_class['delivery date'].dt.day
    df_class['delivery date_month'] = df_class['delivery date'].dt.month
    df_class['delivery date_year'] = df_class['delivery date'].dt.year
    df_class["quantity_kg_log"] = np.log(df_class['quantity_kg'])
    df_class['selling_price_log'] = np.log(df_class['selling_price'])
    df_class['thickness_log'] = np.log(df_class['thickness'])

    # droping item date and delivery date columns hence we converted to day,month,year columns separated 
    df1_class = df_class.copy()
    
    df2_class = df1_class.drop(["item_date","delivery date","quantity tons",'thickness','selling_price','quantity_kg'],axis=1)

    df3_class = df2_class[["country","application","width","quantity_kg_log","selling_price_log","thickness_log","delivery_days","item_date_day","item_date_month","item_date_year","delivery date_day","delivery date_month","delivery date_year","item type_IPL","item type_Others","item type_PL","item type_S","item type_SLAWR","item type_W","item type_WI"]]

    model = joblib.load("model_class.joblib")
    scaler = joblib.load("scaler.joblib")
    
    if df3_class.isnull().values.any():
        
        col2.container(border=True).markdown("***Please enter the input to predict the status***")
        
    elif (df_class["selling_price"]==0).any():
        
        col2.container(border=True).markdown("***Please enter the input to predict the status***")
        
    else:
        scaled_data = scaler.transform(df3_class)
        predictions = model.predict(scaled_data)
        col2.subheader("Predictions")
        col2.container(border=True).text(predictions[0])
        
if selected == "Regression":
    with st.sidebar:
        df_features = pd.read_csv ("ml_features.csv")
        country = st.selectbox("country list",df_features['country'].unique())
        application = st.number_input("application")
        width = st.number_input("width")
        thickness = st.number_input("thickness")
        quantity = st.number_input("quantity tons")
        item_type_IPL= st.selectbox("item type_IPL",[0,1])
        item_type_Others= st.selectbox("item type_Others",[0,1])
        item_type_PL= st.selectbox("item type_PL",[0,1])
        item_type_S= st.selectbox("item type_S",[0,1])
        item_type_SLAWR= st.selectbox("item type_SLAWR",[0,1])
        item_type_W= st.selectbox("item type_W",[0,1])
        item_type_WI= st.selectbox("item type_WI",[0,1])
        
        item_date = st.date_input("item date",value=None)
        delivery_date = st.date_input("Delivery date",value=None)
        
    regression = { 
                        "country":  country,
                        "application": application,
                        "width": width,
                        "thickness": thickness,
                        "quantity tons": quantity,
                        "item type_IPL": item_type_IPL,
                        "item type_Others": item_type_Others,
                        "item type_PL":item_type_PL,
                        "item type_S":item_type_S,
                        "item type_SLAWR":item_type_SLAWR,
                        "item type_W": item_type_W,
                        "item type_WI":item_type_WI,
                        "item_date": item_date,
                        "delivery date": delivery_date                  
                        }
        
    df_class = pd.DataFrame([regression]) 
    
    col1,col2 = st.columns(2)
    
    col1.markdown("## Input values")
    col1.table(df_class.T)

    df_class["item_date"] = pd.to_datetime(df_class["item_date"],format='%Y%m%d')
    df_class["delivery date"] = pd.to_datetime(df_class["delivery date"],format='%Y%m%d')
    
    df_class["quantity_kg"] = df_class["quantity tons"]*1000
    df_class["delivery_days"] = (df_class["delivery date"] - df_class["item_date"]).dt.days
    df_class['item_date_day'] = df_class['item_date'].dt.day
    df_class['item_date_month'] = df_class['item_date'].dt.month
    df_class['item_date_year'] = df_class['item_date'].dt.year
    df_class['delivery date_day'] = df_class['delivery date'].dt.day
    df_class['delivery date_month'] = df_class['delivery date'].dt.month
    df_class['delivery date_year'] = df_class['delivery date'].dt.year
    df_class["quantity_kg_log"] = np.log(df_class['quantity_kg'])
    df_class['thickness_log'] = np.log(df_class['thickness'])

    # droping item date and delivery date columns hence we converted to day,month,year columns separated 
    df1_class = df_class.copy()
    
    df2_class = df1_class.drop(["item_date","delivery date","quantity tons",'thickness','quantity_kg'],axis=1)

    df3_class = df2_class[["country","application","width","quantity_kg_log","thickness_log","delivery_days","item_date_day","item_date_month","item_date_year","delivery date_day","delivery date_month","delivery date_year","item type_IPL","item type_Others","item type_PL","item type_S","item type_SLAWR","item type_W","item type_WI"]]

    model = joblib.load("model_regression.joblib")
    scaler_reg = joblib.load("scaler_reg.joblib")

    col2.subheader("Predictions")
    
    if df3_class.isnull().values.any():
        
        col2.container(border=True).markdown("***Please enter the input to predict the selling price***")
        
    else:        
        scaled_data = scaler_reg.transform(df3_class)
        
        predictions = model.predict(scaled_data)
        predictions_exp_log = np.exp(predictions)
        col2.container(border=True).text(predictions_exp_log[0])