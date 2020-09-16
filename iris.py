#import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn 
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle

#load dataset
data = pd.read_csv("iris.csv")

#set backround image 
page_bg_img = '''
<style>
body {
background-image: url("https://image.freepik.com/free-photo/pink-powder-explosion-white-background-pink-dust-splash-cloud-launched-colorful-parti_36326-37.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Building a Machine Learning Model 

# spliting data for Training and Testing where 'x' training data and 'y' is testing data
x =data.iloc[ : , :4]
y =data['species']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=30,random_state=3)

#fit and predict model
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
y_prd=model.predict(x_test)

#create pickle file
pickle_out=open("predict.pkl",'wb')
pickle.dump(model,pickle_out)
pickle_out.close()

pickle_in=open("predict.pkl",'rb')
predicter = pickle.load(pickle_in)



#set title and markdown
st.title("Classification of IRIS Flower ")
st.markdown("- Their are three species of Iris (Iris setosa, Iris virginica and Iris versicolor)")
st.markdown("- Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters")
st.markdown("- Based on the combination of these four features,developed a model to distinguish the species from each other.")

#add image
st.markdown("![IRIS SPECIES : ](https://miro.medium.com/max/1000/1*Hh53mOF4Xy4eORjLilKOwA.png)")

#add sidebar and write markdown
st.sidebar.markdown("Classify IRIS flower specie according to values you have: ")

#find minimun and maximun values of the length and the width of the sepals and petals
plmax = data['petal_length'].max()
plmin = data['petal_length'].min()

pwmax = data['petal_width'].max()
pwmin = data['petal_width'].min()

slmax = data['sepal_length'].max()
slmin = data['sepal_length'].min()

swmax = data['sepal_width'].max()
swmin = data['sepal_width'].min()

#by using min and max values of the length and the width of the sepals and petals create slider for each feature

pl = st.sidebar.slider("Value of Petal Length",float(plmin),float(plmax))
pw = st.sidebar.slider("Value of Petal Length",float(pwmin),float(pwmax))
sl = st.sidebar.slider("Value of Petal Length",float(slmin),float(slmax))
sw = st.sidebar.slider("Value of Petal Length",float(swmin),float(swmax))

#print("According to data flower specie is :",x)

#add success bar 
if st.sidebar.button(" Classify "):
    x=model.predict([[sl,sw,pl,pw]])
    st.sidebar.success("IRIS Flower Specie is : {} ".format(x))
