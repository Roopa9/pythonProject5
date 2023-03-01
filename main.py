# This is a sample Python script.
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
pio.templates.default = "plotly_white"
data = pd.read_csv("C:\\Users\\Public\\ads.csv")
print(data.head())
data["Clicked on Ad"] = data["Clicked on Ad"].map({0: "No",
                               1: "Yes"})
#Click Through Rate Analysis
#Now let’s analyze the click-through rate based on the time spent by the users on the website:1

fig = px.box(data,
             x="Daily Time Spent on Site",
             color="Clicked on Ad",
             title="Click Through Rate based Time Spent on Site",
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()
fig = px.box(data,
             x="Daily Internet Usage",
             color="Clicked on Ad",
             title="Click Through Rate based on Daily Internet Usage",
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()
fig = px.box(data,
             x="Age",
             color="Clicked on Ad",
             title="Click Through Rate based on Age",
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()
fig = px.box(data,
             x="Area Income",
             color="Clicked on Ad",
             title="Click Through Rate based on Income",
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()
#Calculating CTR of Ads
#Now let’s calculate the overall Ads click-through rate. Here we need to calculate the ratio of users who clicked on the ad to users who left an impression on the ad. So let’s see the distribution of users:


data["Clicked on Ad"].value_counts()
click_through_rate = 4917 / 10000 * 100
print(click_through_rate)
data["Gender"] = data["Gender"].map({"Male": 1,
                               "Female": 0})
x=data.iloc[:,0:7]
x=x.drop(['Ad Topic Line','City'],axis=1)
y=data.iloc[:,9]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                           test_size=0.2,
                                           random_state=4)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x, y)
from sklearn.metrics import accuracy_score
y_pred = model.predict(xtest)
print(accuracy_score(ytest,y_pred))
print("Ads Click Through Rate Prediction : ")
a = float(input("Daily Time Spent on Site: "))
b = float(input("Age: "))
c = float(input("Area Income: "))
d = float(input("Daily Internet Usage: "))
e = input("Gender (Male = 1, Female = 0) : ")

features = np.array([[a, b, c, d, e]])
print("Will the user click on ad = ", model.predict(features))

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Define the streamlit app
def app():
    st.title('Ad Click Prediction')
    st.write('Enter customer data to predict if they will click on an ad:')
    age = st.slider('Age', 18, 65, 25)
    daily_time_spent = st.slider('Daily Time Spent on Site (in minutes)', 0, 100, 50)
    income = st.slider('Estimated Salary (in thousands)', 10, 200, 50)
    country = st.selectbox('Country', data['Country'].unique())


    # Make predictions
    prediction = clf.predict([[age, daily_time_spent, income, country]])

    # Show the results
    if prediction[0] == 0:
        st.write('The customer is not likely to click on an ad.')
    else:
        st.write('The customer is likely to click on an ad.')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
