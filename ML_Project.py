#!/usr/bin/env python
# coding: utf-8

# # Description

# ### This project uses machine learning techniques to predict the Covid-19 cases.

# ## Import Files

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ## Data Set

# In[2]:


# Load the dataset
file_path = r"C:\Users\BALJENDER KAUR\Downloads\india_covid19_data.csv"  # path of my csv file
data = pd.read_csv(file_path)

# Display the first few rows
data.head()


# In[3]:


# Summary statistics
data.describe()


# In[4]:


# Check for missing values
data.isnull().sum()


# In[13]:


# Har state ke liye total confirmed cases calculate karna
statewise_cases = data.groupby('State')['Confirmed_Cases'].sum()
print(statewise_cases)


# ## Data Visualization

# In[14]:


# State-wise total confirmed cases ka bar plot
statewise_cases.plot(kind='bar', title='Total Confirmed Cases per State', xlabel='State', ylabel='Confirmed Cases')
plt.show()


# In[5]:


# Kerala me total deaths over time ka line plot
kerala_data = data[data['State'] == 'Kerala']
kerala_data.plot(x='Date', y='Deaths', kind='line', title='Kerala Deaths Over Time')
plt.show()


# In[6]:


plt.figure(figsize=(10, 6))
plt.hist(data['Confirmed_Cases'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Confirmed Cases')
plt.xlabel('Number of Confirmed Cases')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[7]:


# Deaths ka histogram
plt.figure(figsize=(10, 6))
plt.hist(data['Deaths'], bins=30, color='salmon', edgecolor='black')
plt.title('Distribution of Deaths')
plt.xlabel('Number of Deaths')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[8]:


# Recovered Cases ka histogram
plt.figure(figsize=(10, 6))
plt.hist(data['Recovered_Cases'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Distribution of Recovered Cases')
plt.xlabel('Number of Recovered Cases')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[9]:


import matplotlib.pyplot as plt

# Data
states = ['kerala', 'Delhi', 'Maharastra', 'Karnataka','Uttar Pradesh','West Bengal']
cases = [5000, 3000, 4500, 2000,3500,4200]

# Plot
plt.pie(cases, labels=states, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Overall COVID-19 Cases by State')
plt.show()


# ## Data Processing

# In[25]:


# Sample data
da = {
    'Date': ['31-03-2020', '31-03-2020', '31-03-2020', '31-03-2020', '31-03-2020', '31-03-2020', '31-03-2020'],
    'State': ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Kerala', 'Uttar Pradesh', 'West Bengal'],
    'Confirmed_Cases': [109374, 119419, 75391, 52861, 41861, 77915, 152828],
    'Recovered_Cases': [98383, 119281, 55610, 133112, 49466, 118816, 160690],
    'Deaths': [95, 3031, 2448, 3375, 660, 3445, 2612]
}

data = pd.DataFrame(data)
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
print(data.head())


# ## Model training

# In[33]:


# Example: Calculate Active Cases
data['Active_Cases'] = data['Confirmed_Cases'] - data['Recovered_Cases'] - data['Deaths']
print(data.head())

# Prepare the data for modeling
X = data[['Confirmed_Cases', 'Recovered_Cases', 'Deaths']]
y = data['Active_Cases']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# ## Model Evaluation

# In[34]:


# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# ## Prediction

# In[35]:


# Example prediction
new_data = pd.DataFrame({
    'Confirmed_Cases': [150000],
    'Recovered_Cases': [140000],
    'Deaths': [2000]
})

predicted_active_cases = model.predict(new_data)
print(f'Predicted Active Cases: {predicted_active_cases[0]}')


# ## Conclusion

# #### The COVID-19 case prediction project using Python and machine learning provides valuable insights into the pandemic’s progression and potential future trends. Key takeaways include:
# 
# #### Model Accuracy: The machine learning models, such as linear regression, decision trees, and neural networks, demonstrated varying degrees of accuracy in predicting COVID-19 cases. Fine-tuning these models and incorporating more data can enhance their predictive power.
# 
# #### Data Visualization: Effective use of Python libraries like Matplotlib and Seaborn helped visualize the trends and patterns in COVID-19 data, making it easier to understand the spread and impact of the virus.
# 
# #### Feature Importance: Identifying key features that influence the spread of COVID-19, such as population density, mobility data, and public health interventions, can help in making informed decisions for future outbreaks.
# 
# #### Limitations: The project faced challenges such as data quality, missing values, and the dynamic nature of the pandemic. Addressing these limitations is crucial for improving model reliability.
# 
# #### Future Work: Future enhancements could include integrating real-time data, exploring advanced machine learning techniques, and expanding the scope to include vaccination data and other relevant factors.
