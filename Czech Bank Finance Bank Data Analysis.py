#!/usr/bin/env python
# coding: utf-8

# # Czechoslovakia Bank Financial Data Analysis

# In[ ]:





# In[ ]:


#import the libraries to amnipulate the data


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import date


# In[2]:


#import the data from the specified folder
accounts_df = pd.read_csv ('D:/credit/account.csv',sep=";")
cards_df = pd.read_csv ('D:/credit/card.csv', sep = ';')
clients_df = pd.read_csv ('D:/credit/client.csv', sep = ';')
dispos_df = pd.read_csv ('D:/credit/disp.csv', sep = ';')
district_df = pd.read_csv ('D:/credit/district.csv', sep = ';')
loan_df = pd.read_csv ('D:/credit/loan.csv', sep = ';')
order_df = pd.read_csv ('D:/credit/order.csv', sep = ';')
trans_df = pd.read_csv ('D:/credit/trans.csv', sep = ';')


# In[3]:


files = [accounts_df, cards_df, clients_df, dispos_df, district_df, loan_df, order_df, trans_df]
files_name = ['accounts_df', 'cards_df', 'clients_df', 'dispos_df', 'district_df', 'loan_df', 'order_df', 'trans_df']


# In[4]:


for id, item in enumerate (files): 
    print('Dataframe name: ' + str(files_name [id]) + " with number of rows:" + str(item.shape[0]) + ' and columns:' + str(item.shape[1]))
    display(item.describe())
    print(item.isnull().sum())
    print('\n')


# In[5]:


for id, item in enumerate(files): 
    print('Dataframe name:' + str(files_name [id]))
    display(item.head (n=3))
    print('\n')


# In[6]:


#Create a list that have a worng date column to cahnge the date dd-mm-yyyy pattern

date_cor_files = [trans_df, accounts_df, loan_df]


# In[7]:


def date_correction (df, col_name):
    df [col_name] = pd.to_datetime (df[col_name], format = '%y%m%d', errors = 'coerce')
    return df


# In[8]:


for id, item in enumerate (date_cor_files): 
    date_cor_files[id] = date_correction(item, 'date')

trans_df = date_cor_files[0]
accounts_df = date_cor_files[1]
loans_df = date_cor_files[2]


# In[9]:


for id, item in enumerate(files): 
    print('Dataframe name:' + str(files_name [id]))
    display(item.head (n=3))
    print('\n')


# In[ ]:


#change the district data column from Abrevation to required and effcient name


# In[10]:


district_df=district_df.rename(columns={"A1":"district_id",
                                        "A2":"District name",
                                        "A3":"Region",
                                        "A4":"No. of inhabitants",
                                        "A5":"No. of municipalities with inhabitants < 499",
                                        "A6":"No. of municipalities with inhabitants 500-1999",
                                        "A7":"No. of municipalities with inhabitants 2000-9999",
                                        "A8":"No. of municipalities with inhabitants >10000",
                                        "A9":"No. of cities",
                                        "A10":"Ratio of urban inhabitants",
                                        "A11":"Average salary",
                                        "A12":"Unemployment rate '95",
                                        "A13":"Unemployment rate '96",
                                        "A14":"No. of entrepreneurs per 1000 inhabitants",
                                        "A15":"no. of committed crimes '95",
                                        "A16":"no. of committed crimes '96"
                                    
                                       
                                       
                                       })


# In[11]:


district_df


# In[32]:


clients_df.head()


# In[ ]:





# # What is the demographic profile of the bank's clients and how does it vary across districts?

# In[ ]:


# for that question we need to add the both data are client data and district data to check the client district profile


# In[ ]:


#Before merge we need to change the birthnumber to birthdate and get age and gender to visualize and analyze the data


# In[22]:


client_data=clients_df.copy()


# In[23]:


client_data.head()


# In[34]:


client_data = client_data.join (pd.DataFrame ( { 'birth_date': np.nan, 'gender': np.nan, 'age': np.nan}, index = clients_df.index))


# In[35]:


client_data['birth_date'] = client_data['birth_number']
for ids, item in enumerate (client_data['birth_number']):
    if int (str (item) [2:4]) > 50:
        client_data.loc [ids, 'gender'] = 0 #female
        client_data.loc [ids, 'birth_date'] = item - 5000 
    else: 
        client_data.loc [ids, 'gender'] = 1 


# In[46]:


mask = client_data["birth_date"].dt.year > 2000
client_data.loc[mask, "birth_date"] = client_data.loc[mask, "birth_date"] - pd.DateOffset(years=100)


# In[49]:


client_data


# In[39]:


client_data["birth_date"]=pd.to_datetime(client_data["birth_date"], format = '%y%m%d', errors = 'coerce')


# In[48]:


client_data["age"]=1999-client_data["birth_date"].dt.year


# In[50]:


district_data=district_df.copy()#Copy the data because to if we manipulate the data original data remain same 


# In[51]:


clients_district = pd.merge(client_data, district_data, on='district_id')


# In[54]:


clients_district.head()


# In[53]:


clients_district["gender"]=clients_district["gender"].map( {0: 'Female', 1: 'Male'})


# In[55]:


# Count the number of clients in each district
district_client_count = clients_district['District name'].value_counts()

# Plot the distribution of clients across districts
plt.figure(figsize=(10, 6))
district_client_count.plot(kind='bar')
plt.title('Distribution of Clients Across Districts')
plt.xlabel('District')
plt.ylabel('Number of Clients')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[57]:


# Plot the age distribution of clients for each district
plt.figure(figsize=(10, 6))
clients_district.groupby('District name')['age'].plot(kind='hist', alpha=0.5, bins=20, legend=True)
plt.title('Age Distribution of Clients Across Districts')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend(title='District')
plt.tight_layout()
plt.show()


# In[58]:


#Ditrict average salary 
plt.figure(figsize=(10, 6))
clients_district.groupby('District name')['Average salary'].mean().plot(kind='bar', color='skyblue')
plt.title('Average Salary Across Districts')
plt.xlabel('District')
plt.ylabel('Average Salary')
plt.legend(title='District')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[59]:


# Plot the unemployment rate across districts
plt.figure(figsize=(10, 6))
clients_district.groupby('District name')['Unemployment rate \'96'].mean().plot(kind='bar', color='salmon')
plt.title('Unemployment Rate \'96 Across Districts')
plt.xlabel('District')
plt.ylabel('Unemployment Rate')
plt.legend(title='District')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[62]:


#Ditrict average salary 
plt.figure(figsize=(10, 6))
clients_district.groupby('Region')['Average salary'].mean().plot(kind='bar', color='skyblue')
plt.title('Average Salary Across Region')
plt.xlabel('Region')
plt.ylabel('Average Salary')
plt.legend(title='District')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# #Business Question-What is the demographic profile of the bank's clients and how does it vary across districts?
# 
# We extract the date,age and gender through client data.To view the required insight to analyze the data.
# To perform a bank client and district data we need to merge the data through district id and get the personalyzed data.
# 
# Through Data:
#     We check the 1)Distribution of Clients Across Districts
#                  2)Age Distribution of Clients Across Districts
#                  3)Average Salary Across Districts
#                  4)Unemployment Rate \96 Across Districts
#                  5)Average Salary Across Region

# In[ ]:





# # How the banks have performed over the years. Give their detailed analysis year & month-wise.

# In[ ]:





# In[63]:


trans_df.head()


# In[64]:


trans_data=trans_df.copy()


# In[ ]:





# In[65]:


# Convert 'date' column to datetime format
trans_data['date'] = pd.to_datetime(trans_data['date'])

# Extract year and month from the date column
trans_data['year'] = trans_data['date'].dt.year
trans_data['month'] = trans_data['date'].dt.month

# Group by year and month and calculate financial metrics
bank_performance = trans_data.groupby(['year', 'month']).agg({
    'amount': 'sum',        # Total amount transacted
    'balance': 'mean',      # Average account balance
    'trans_id': 'count'     # Number of transactions
}).reset_index()



# In[66]:


# Plotting the bank performance metrics
plt.figure(figsize=(12, 8))


plt.plot(bank_performance['amount'], marker='o', color='blue')
plt.title('Total Amount Transacted Over Time')
plt.xlabel('Time')
plt.ylabel('Total Amount')
plt.tight_layout()
plt.show()


# In[67]:


plt.figure(figsize=(12, 8))
plt.plot(bank_performance['balance'], marker='o', color='green')
plt.title('Average Account Balance Over Time')
plt.xlabel('Time')
plt.ylabel('Average Balance')
plt.tight_layout()
plt.show()


# In[68]:


plt.figure(figsize=(12, 8))
plt.plot(bank_performance['trans_id'], marker='o', color='orange')
plt.title('Number of Transactions Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Transactions')

plt.tight_layout()
plt.show()


# In[69]:


#Bank Transaction happend in every Year
x=trans_data["year"]
y=trans_data["year"].value_counts()


# In[70]:


#yearly plot

plt.plot(y)
plt.xlabel("Years")
plt.ylabel("Values")
plt.title("No of transaction happend in the every year")
plt.show()


# In[71]:


month_codes = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sep',
        10: 'Oct',
        11: 'Nov',
        12: 'Dec'
    }


# In[72]:


trans_data["month"] = trans_data["month"].map(month_codes)


# In[74]:


#Bank Transaction happend in every month
x=trans_data["month"]
y=trans_data["month"].value_counts()


# In[76]:


#monthly plot
trans_data["month"].value_counts().plot(kind="bar")
plt.xlabel("month")
plt.ylabel("Values")
plt.title("No of transaction happend in the every month")
plt.show()


# In[77]:


#Monthly plot transction shown in number
plt.plot(y)
plt.xlabel("month")
plt.ylabel("Values")
for i, v in enumerate(y):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.title("No of transaction happend in the every month")
plt.show()


# In[78]:


#yearly amount 
x=trans_data["year"]
y=trans_data["amount"].sum()


# In[79]:


data={"year":x,"Amount sum":y}


# In[80]:


df = pd.DataFrame(data)

# Aggregate the data by year and sum the amounts
df_agg = df.groupby('year')['Amount sum'].sum().reset_index()

# Plot the aggregated data
sns.barplot(x='year', y='Amount sum', data=df_agg)
plt.xlabel("Year")
plt.ylabel("Total Amount")
plt.title("Total Amount per Year")
plt.show()


# In[81]:


# Plot the aggregated data
sns.lineplot(x='year', y='Amount sum', data=df_agg)
plt.xlabel("Year")
plt.ylabel("Total Amount")
plt.title("Total Amount per Year")
plt.show()


# In[82]:


#total amount sum in every month
x=trans_data["month"]
y=trans_data["amount"].sum()


# In[83]:


data={"month":x,"Amount sum":y}


# In[84]:


df = pd.DataFrame(data)

# Aggregate the data by year and sum the amounts
df_agg = df.groupby('month')['Amount sum'].sum().reset_index()

# Plot the aggregated data
sns.barplot(x='month', y='Amount sum', data=df_agg)
plt.xlabel("month")
plt.ylabel("Total Amount")
plt.title("Total Amount per month")
plt.show()


# In[85]:


sns.lineplot(x='month', y='Amount sum', data=df_agg)
plt.xlabel("month")
plt.ylabel("Total Amount")
plt.title("Total Amount per month")
plt.show()


# In[86]:


#Check the balance on every year

x=trans_data["year"]
y=trans_data["balance"].sum()


# In[87]:


data={"year":x,"Balance sum":y}


# In[88]:


df = pd.DataFrame(data)

# Aggregate the data by year and sum the amounts
df_agg = df.groupby('year')['Balance sum'].sum().reset_index()

# Plot the aggregated data
sns.barplot(x='year', y='Balance sum', data=df_agg)
plt.xlabel("Year")
plt.ylabel("Total Balance")
plt.title("Total Balance per month")
plt.show()


# In[89]:


#month amount

x=trans_data["month"]
y=trans_data["balance"].sum()


# In[90]:


data={"month":x,"Balance sum":y}


# In[91]:


df = pd.DataFrame(data)

# Aggregate the data by year and sum the amounts
df_agg = df.groupby('month')['Balance sum'].sum().reset_index()

# Plot the aggregated data
sns.barplot(x='month', y='Balance sum', data=df_agg)
plt.xlabel("month")
plt.ylabel("Total Balance")
plt.title("Total Balance per month")
plt.show()


# In[92]:


trans_data["date"].min()


# In[93]:


trans_data["date"].max()


# In[94]:


trans_data["amount"].sum()


# In[95]:


trans_data["amount"].mean()


# In[96]:


trans_data["balance"].sum()


# In[97]:


trans_data["balance"].mean()


# #Business Question-How the banks have performed over the years. Give their detailed analysis year & month-wise?
# 
# We did the sum,mean of every month,every year of amount and balance in the data.
# 
# Through Data:
#     We check the 1)Month-amount and Year-mean plot to chect the min and max values data month and year through data plot.
#                  2)Month-balance and Year-balance plot to chect the min and max values data month and year through data plot.
#                  3)We generate a timestrap plot to check the performance of detailed analysis of year and month.
#                  4)We get the mean amount result 5924.145 and sum is 6257793560.300002.
#                  5)We get the mean balance result 38518.33 and sum is 40687683193.99999.

# In[ ]:





# # What are the most common types of accounts and how do they differ in terms of usage and profitability?

# In[98]:


account_data=accounts_df.copy()


# In[99]:


trans_dataa=trans_df.copy()


# In[100]:


account_trans = pd.merge(account_data, trans_dataa, on='account_id')


# In[101]:


account_trans.head()


# In[102]:


account_profitability = account_trans.groupby('frequency').agg({
    'amount': 'sum',        # Total transaction amount
    'balance': 'mean'      # Average account balance
})

# Plotting profitability metrics
plt.figure(figsize=(10, 6))
account_profitability.plot(kind='bar', rot=0)
plt.title('Profitability by Account Type')
plt.xlabel('Account Type')
plt.ylabel('Value')
plt.legend(['Total Transaction Amount', 'Average Balance'])
plt.tight_layout()
plt.show()


# In[103]:


a=account_trans["frequency"].value_counts()
b=account_trans["type"].value_counts()
c=account_trans["operation"].value_counts()


# In[104]:


a,b,c


# In[105]:


# Plot pie charts for each dataset
plt.figure(figsize=(15, 5))
plt.pie(a, labels=a.index, autopct='%1.1f%%')
plt.title('Account Trans frequency Percentage')
plt.tight_layout()
plt.show()


# In[106]:


plt.pie(b, labels=b.index, autopct='%1.1f%%')
plt.title('Account Trans Type Percentage')


# In[107]:


plt.pie(c, labels=c.index, autopct='%1.1f%%')
plt.title('Account Trans operation Percentage')


# #Bussiness Question-What are the most common types of accounts and how do they differ in terms of usage and profitability?
# 
# We check the transaction of frequency percentage,Type,operation percentage to check their performance impact on profitability
# 
# From data we have :
#           A)"Account Trans frequency Percentage" of frequency of issuance of statements: "POPLATEK MESICNE" stands for monthly               issuance is 91.8%."POPLATEK TYDNE" stands for weekly issuance is 5.9%. "POPLATEK PO OBRATU" stands for issuance                 after transaction is 2.3%
#           B)"Account Trans Type Percentage" VYDAJ is 60.1% Withdrawl and Prijem is 38.3% and VYBER 1.6% withdrawl in cash.
#           C)"Account Trans operation Percentage" of mode of transaction: "VYBER KARTOU" credit card withdrawal is 0.9%, "VKLAD"             credit in cash is 18% ,"PREVOD Z UCTU" collection from another bank  is 7.5%,"VYBER" withdrawal in cash is 49.8%,                "PREVOD NA UCET" remittance to another bank is 23.9%.

# In[ ]:





# In[ ]:


#Which types of cards are most frequently used by the bank's clients and what is the overall profitability of the credit card business?


# In[108]:


cards_data=cards_df.copy()


# In[109]:


dispos_data=dispos_df.copy()


# In[110]:


card_clients = pd.merge(cards_data, dispos_data, on='disp_id')


# In[111]:


card_clients.head()


# In[112]:


a=card_clients["type_x"].value_counts()


# In[113]:


a.plot(kind='bar', subplots=True)
plt.title('Credit Card Analysis')
plt.xlabel('Card Type')
plt.ylabel('Credit Card Count')
plt.show()


# In[114]:


card_combine_data=pd.merge(card_clients,trans_data[["account_id","trans_id","amount","balance"]] ,on='account_id', how='left')


# In[115]:


card_combine_data.head()


# In[116]:


# Calculate sum and mean of amounts for each card type
classic_sum = card_combine_data[card_combine_data['type_x'] == 'classic']['amount'].sum()
classic_mean = card_combine_data[card_combine_data['type_x'] == 'classic']['amount'].mean()

# Print results for classic card type
print("Classic Card Type:")
print("Sum of Amounts:", classic_sum)
print("Mean of Amounts:", classic_mean)


# In[117]:


# Calculate sum and mean of amounts for each card type
junior_sum = card_combine_data[card_combine_data['type_x'] == 'junior']['amount'].sum()
junior_mean = card_combine_data[card_combine_data['type_x'] == 'junior']['amount'].mean()

# Print results for classic card type
print("Junior Card Type:")
print("Sum of Amounts:", junior_sum)
print("Mean of Amounts:", junior_mean)


# In[118]:


# Calculate sum and mean of amounts for each card type
gold_sum = card_combine_data[card_combine_data['type_x'] == 'gold']['amount'].sum()
gold_mean = card_combine_data[card_combine_data['type_x'] == 'gold']['amount'].mean()

# Print results for classic card type
print("Gold Card Type:")
print("Sum of Amounts:", gold_sum)
print("Mean of Amounts:", gold_mean)


# In[119]:


sum_of_data=[classic_sum,junior_sum,gold_sum]

mean_of_data=[classic_mean,junior_mean,gold_mean]


# In[120]:


card_types = ['Classic', 'Junior', 'Gold']

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot for sum
ax[0].bar(card_types, sum_of_data, color='skyblue')
ax[0].set_title('Sum of Amount by Card Type')
ax[0].set_xlabel('Card Type')
ax[0].set_ylabel('Sum of Amount')

# Plot for mean
ax[1].bar(card_types, mean_of_data, color='lightgreen')
ax[1].set_title('Mean of Amount by Card Type')
ax[1].set_xlabel('Card Type')
ax[1].set_ylabel('Mean of Amount')

plt.tight_layout()
plt.show()


# In[124]:


card_type_profitability = card_combine_data.groupby('type_x')['amount'].sum()


# In[125]:


overall_profitability = card_type_profitability.sum()


# In[126]:


overall_profitability


# #Which types of cards are most frequently used by the bank's clients and what is the overall profitability of the credit card business?
# 
# In this data question to show the distribution of cards types percentage,sum and mean of all credit cards types.
# We got the overall_profitability sum of all credit card total amount.

# In[ ]:





# # What are the major expenses of the bank and how can they be reduced to improve profitability?

# In[ ]:





# In[127]:


trans_data=trans_df.copy()


# In[128]:


trans_data.head()


# In[130]:


trans_data["k_symbol"].value_counts()


# In[133]:


trans_data['k_symbol'].replace({" ": "Null"}, inplace=True)


# In[134]:


trans_data["k_symbol"].value_counts()


# In[135]:


#Trans Exp bank in each year
trans_data['year'] = pd.to_datetime(trans_data['date']).dt.year


# In[139]:


# Group the DataFrame by year and transaction type, and calculate total amount for each type in each year
exp_summ_data_year=trans_data.groupby(['year', 'type'])['amount'].sum().unstack()
# Calculate the percentage of each transaction type for each year
expense_summary_percentage = exp_summ_data_year.div(exp_summ_data_year.sum(axis=1), axis=0) * 100


# In[140]:


# Plotting the percentage plot
plt.figure(figsize=(10, 6))
expense_summary_percentage.plot(kind='bar', stacked=True)
plt.title('Percentage of Credit and Withdrawal Amounts by Year')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Transaction Type')
plt.tight_layout()
plt.show()


# In[141]:


#Trans Exp bank in each operation
expense_trans_oper = trans_data[trans_data['operation'].notnull()]

# Group expense transactions by operation and calculate total expenses for each operation
operation_expenses = expense_trans_oper.groupby('operation')['amount'].sum().sort_values(ascending=False)

# Display operation expenses data
print(operation_expenses)


# In[142]:


# Calculate the total expenses for all operations
total_expenses = operation_expenses.sum()

# Calculate the percentage of each operation
operation_percentages = (operation_expenses / total_expenses) * 100

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(operation_expenses, labels=operation_expenses.index, autopct='%1.1f%%', startangle=140)
plt.title('Operation Expenses Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[143]:


#Trans Exp bank in each k_symbol amount percentage 
# Assuming 'trans_data' is your transaction DataFrame

# Filter transaction data to include only expense transactions
expense_transactions = trans_data[trans_data['operation'].notnull()]

# Group expense transactions by 'k_symbol' and calculate count and sum
k_symbol_expenses = expense_transactions.groupby('k_symbol')['amount'].agg(['count', 'sum'])

# Calculate total expenses
total_expenses = k_symbol_expenses['sum'].sum()

# Calculate percentage of expenses for each k_symbol
k_symbol_expenses['percentage'] = (k_symbol_expenses['sum'] / total_expenses) * 100


# In[144]:


# Plot the bar graph
plt.figure(figsize=(12, 6))
plt.bar(k_symbol_expenses.index, k_symbol_expenses['sum'], color='orange', alpha=0.7, label='Sum')
plt.ylabel('Count / Sum')
plt.xlabel('k_symbol')
plt.title('Expenses by k_symbol')
plt.legend()
# Add percentage labels on the bars
for i, perc in enumerate(k_symbol_expenses['percentage']):
    plt.text(i, k_symbol_expenses.iloc[i]['sum'] + 100, f'{perc:.2f}%', ha='center')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In above generated code i show the the bank expenses happen through operation,trans,k_symbol:
#     A)Percentage of Credit and Withdrawal Amounts by Year
#     B)Operation Expenses Distribution
#     C)Expenses by k_symbol

# In[ ]:





# # What is the bank’s loan portfolio and how does it vary across different purposes and client segments?

# In[145]:


loan_data=loan_df.copy()


# In[146]:


dispos_data=dispos_df.copy()


# In[147]:


loan_info = pd.merge(loan_data, dispos_data, on='account_id')


# In[148]:


clients_data=clients_df.copy()


# In[149]:


loan_info = pd.merge(loan_info, clients_data, on='client_id')


# In[150]:


loan_info.head()


# In[151]:


loan_info['status_desc'] = loan_info['status']

dict1 =  {'A':'Contract finished, no problem', 
      'B':'Contract finised, loan was not paid',
      'C':'Runing contract, OK so far',
      'D':'Runing contract, client in debt'
     }
loan_info.status_desc = loan_info.status_desc.replace(dict1)

loan_info['status_numeric'] = loan_info['status']

#encoding bad loans as 1 and good ones as -1 
dict2 =  {'A':1, 
      'B':-1,
      'C':1,
      'D':-1
     }


# In[152]:


loan_info.status_numeric = loan_info.status_numeric.replace (dict2)
display(loan_info.head (n=3))


# In[153]:


a=loan_info["type"].value_counts()


# In[154]:


a.plot(kind='bar', subplots=True)
plt.title('Loan Type Analysis')
plt.xlabel('Loan Type')
plt.ylabel('Loan Type Count')
plt.show()


# In[155]:


total=loan_info["amount"].sum()
a=loan_info[loan_info["type"] == "OWNER"]["amount"].sum()
b=loan_info[loan_info["type"] == "DISPONENT"]["amount"].sum()


# In[156]:


labels = ["OWNER", 'DISPONENT']
sizes = [a, b]

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Loan Distribution by Category')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[157]:


status_counts = loan_info["status_numeric"].value_counts()

# Plotting the bar graph
plt.figure(figsize=(8, 6))
status_counts.plot(kind='bar', color='skyblue')
plt.title('Loan Status Distribution')
plt.xlabel('Status Numeric')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels if needed
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[158]:


# Count the occurrences of each unique value in 'status_numeric' column
status_counts = loan_info["status_numeric"].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Loan Status info')
plt.legend(["On Time", "Delayed Time"], title="Loan Status")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[159]:


status_counts = loan_info["status_desc"].value_counts()

# Plotting the bar graph
plt.figure(figsize=(8, 6))
status_counts.plot(kind='bar', color='skyblue')
plt.title('Loan Status Distribution')
plt.xlabel('Status Numeric')
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotate x-axis labels if needed
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[160]:


# Count the occurrences of each unique value in 'status_numeric' column
status_counts = loan_info["status_desc"].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Loan Status info')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[161]:


numerical_columns = ['amount', 'duration', 'payments']


# In[162]:


loan_info['status_numeric'] = pd.to_numeric(loan_info['status_numeric'], errors='coerce')

# Adding 'status_numeric' column to the list of numerical columns
numerical_columns.append('status_numeric')

# Calculate correlation matrix
correlation_matrix = loan_info[numerical_columns].corr()

# Print correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)


# In[163]:


sns.heatmap(correlation_matrix, annot = True, fmt=".2f");


# In[164]:


district_names=[]

for i in range(len(loan_info)):
    district_id = loan_info.loc[i, "district_id"]
    district_name = district_df[district_df["district_id"] == district_id]["District name"].values
    if len(district_name) > 0:
        district_names.append(district_name[0])
    else:
        # If no corresponding district name is found, append "Nan"
        district_names.append("Nan")


# In[165]:


loan_info["district_name"]=district_names


# In[166]:


loan_info["district_name"].value_counts()


# In[167]:


# Group loans by district and calculate the number of loans for each district
district_loan_counts = loan_info['district_name'].value_counts()

# Calculate the percentage of loans for each district
total_loans = district_loan_counts.sum()
district_loan_percentages = (district_loan_counts / total_loans) * 100

# Plotting percentage view of loan portfolio by district
plt.figure(figsize=(12, 8))
district_loan_percentages.plot(kind='bar', color='skyblue')
plt.title('Percentage View of Loan Portfolio by District')
plt.xlabel('District ID')
plt.ylabel('Percentage of Loans')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[168]:


district_loan_percentages.max()


# In[169]:


district_loan_percentages.min()


# In[ ]:





# In[ ]:


Bussiness Question of Loan Data info we merge the data loan,disp,client data for better understanding.

We check and visualize the data through differnt parameters like:
    A)Loan Type Analysis
    B)Loan Distribution by Category
    C)Loan Status Distribution
    D)To check the time status of Loan Status.
    E)Loan Status Distribution on different category.

Above visualization provide the better view insight to analyze the data.


# In[ ]:





# # How can the bank improve its customer service and satisfaction levels?

# Improving customer service and satisfaction levels can be achieved through various strategies based on the insights derived from the data. Here's how you can approach it.

# In[ ]:


Transaction Analysis:


# Through Transaction data to find the trends based on year,month on amount and balance the result shows that every year the transaction increase year by year.
# Providing the better connectivity and knowledge on saving can cost the better transaction result.

# In[ ]:


Analyze Customer Feedback:

If available,analyze any customer feedback or complaints data to identify common issues and pain points.This could include complaints related to account management, transaction processing, or customer support.
# Improving Communication Channels:

# Evaluate the effectiveness of existing communication channels for example phone support, email, chat. Consider implementing new channels or improving existing ones to provide faster and more efficient customer support.

# In[ ]:


Personalized Recommendations:


# Utilize transaction history to offer personalized recommendations to customers. This could include suggesting relevant banking products or services based on their transaction patterns and financial needs.

# In[ ]:


Community Engagement:


# Engage with the community through events, workshops, or social media to build trust and strengthen relationships with customers. This could include financial literacy workshops or community outreach programs.

# In[ ]:





# # Can the bank introduce new financial products or services to attract more customers and increase profitability?

# Here are my top picks for bank can introduce new financial products or services to attract more customers:

# Yes,Bank can introduce new financial products or services:
#     A)House Loan for compounding interest.
#     B)Providing Platinum credit card for high budget payment.
#     C)Business and Entrepeneur Bond for equity percent share.
#     D)Gold Loan for simple interest.
#     E)Health Beneficiary insurance and loan.

# In[ ]:





# In[ ]:


Here how bank can increase profitability on financial products or services.
A)Due interest on loan amount
B)More than 50 transaction on every month can charge 2% in total sum of amount transferd.
C)Minimun Balance Account
D)Credit Card Charges


# In[174]:


loan_data=loan_df.copy()


# In[176]:


loan_data.head()


# In[177]:


finished_loans = loan_data[loan_data['status'] == 'B']
finished_loans['due_percentage'] = 0.05  # 5% due percentage for finished loans
finished_loans['due_amount'] = finished_loans['amount'] * finished_loans['due_percentage']

# Display the result
print("Loans with finished contracts (status 'B'):")
print(finished_loans[['loan_id', 'amount', 'due_percentage', 'due_amount']])


# In[182]:


trans_dataaa=trans_df.copy()


# In[183]:


trans_dataaa.head()


# In[184]:


trans_dataaa['date'] = pd.to_datetime(trans_dataaa['date'])

# Extract month from date
trans_dataaa['month'] = trans_dataaa['date'].dt.month


monthly_summary = trans_dataaa.groupby('month').agg(total_transactions=('trans_id', 'count'), total_amount=('amount', 'sum'))

high_volume_months = monthly_summary[monthly_summary['total_transactions'] > 50]

# Calculate 2% charge for high-volume months
high_volume_months['charge'] = 0.02 * high_volume_months['total_amount']

# Display the result
print("Months with more than 50 transactions and their charge:")
print(high_volume_months)


# These are some financial introduction scheme that can generate the profit for banks.

# In[ ]:





# # Conclusion

# In this python work book i generate python code for manipulate the data to better understanding and visualization to get better result based on the question.

# Here the List of answer we generate the code and visualization.
# 
# A)Demographic profile across through client from various district.
# B)Performance of bank over the years.
# C)Accounts and types how differ from other.
# D)Types of Cards and their results
# E)Loan status info.
# F)Expenses of Bank
# 
# List of answer to improve the bank profitability and performance.

# In[ ]:




