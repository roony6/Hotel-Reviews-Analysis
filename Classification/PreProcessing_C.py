import re
import numpy as np
import pandas as pd
#from ReviewAnalysis import *
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


# Main Function
def ready_data(data, test=False):

    # call encode tags
    data = split_tags(data, 'Tags')

    # Fill Null Values <=> Most Frequent.
    data = fill_nulls(data)

    # Encode The Target Reviewer_Score
    encode_reviewer_score(data)

    # Encode The  Reviewe_Date
    encode_review_date(data)

    # encode columns
    encode_cols = ('Hotel_Name', 'Reviewer_Nationality', 'Hotel_Address', 'T1', 'T2', 'T3', 'T4', 'T5')
    data = feature_encoder(data, encode_cols)
    data = fill_nulls(data)

    # removing 'day' from days_since_review column
    print("Remove Days...")
    regex = re.compile(' days| day')
    data['days_since_review'] = data['days_since_review'].str.replace(regex, '', regex=True)
    data['days_since_review'] = data['days_since_review'].astype(int)

    # encoding negative, positive reviews
    #data = encode_review(data)
    encode_reviews(data, 'Negative_Review', 'Positive_Review')
    data['Negative_Review'] = data['Negative_Review'].astype(int)
    data['Positive_Review'] = data['Positive_Review'].astype(int)

    scale_cols = ['Hotel_Name', 'Reviewer_Nationality', 'Hotel_Address', 'Day', 'Month', 'Year', 'T1', 'T2', 'T3', 'T4',
                  'days_since_review', 'Reviewer_Score', 'Additional_Number_of_Scoring', 'Average_Score',
                  'Review_Total_Negative_Word_Counts', 'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts',
                  'Total_Number_of_Reviews_Reviewer_Has_Given', 'lat', 'lng']
    data = feature_scaling2(data, scale_cols, -1, 1)
    data['T5'] = data['T5'].fillna(data['T5'].mean())

    # dropping columns
    print("Drop Columns...")
    cols = ['Review_Date', 'Tags', 'Reviewer_Score']
    x = data.drop(columns=cols)

    y = data['Reviewer_Score']
    return x, y


def encode_review_date(data):
    print('Encode Reviews Date...')

    if data['Review_Date'].dtype == 'datetime64[ns]':
        data['Review_Date'] = data['Review_Date'].astype(str)
        data[['Day', 'Month', 'Year']] = data['Review_Date'].str.split('-', expand=True)
    else:
        data[['Day', 'Month', 'Year']] = data['Review_Date'].str.split('/', expand=True)

    data[['Day', 'Month', 'Year']] = data[['Day', 'Month', 'Year']].astype(int)

# Encoding Reviews, Method #1
def encode_reviews(review_data, col_neg, col_pos):
    print("Encode Reviews Binary...")
    no_neg = 'No Negative'
    no_positive = 'No Positive'
    review_data[col_neg] = review_data[col_neg].str.replace(no_neg, '0')
    review_data[col_pos] = review_data[col_pos].str.replace(no_positive, '0')
    for row in review_data.itertuples():
        if review_data.at[row.Index, col_neg] != '0':
            review_data.at[row.Index, col_neg] = '1'
        if review_data.at[row.Index, col_pos] != '0':
            review_data.at[row.Index, col_pos] = '1'


# Splitting Tags
def split_tags(data, feature):
    regex = re.compile(" '|'")
    regex2 = re.compile("\[ |\[|\]")
    data[feature] = data[feature].str.replace(regex, '', regex=True)
    data[feature] = data[feature].str.replace(regex2, '', regex=True)

    # Nights Feature
    print("Encode Nights...")
    data['# Nights'] = data[feature].apply(lambda x: re.findall(r'(?:Stayed )(\d+)', x)[0] if re.findall(r'(?:Stayed )(\d+)', x) else 0)

    print("Split Tags...")
    tags = pd.DataFrame(data[feature].str.split(', '))
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []

    for i in tags.itertuples():
        if len(tags.at[i.Index, feature]) > 0 and 'Stayed' in tags.at[i.Index, feature][0]:
            del tags.at[i.Index, feature][0]
        if len(tags.at[i.Index, feature]) > 1 and 'Stayed' in tags.at[i.Index, feature][1]:
            del tags.at[i.Index, feature][1]
        if len(tags.at[i.Index, feature]) > 2 and 'Stayed' in tags.at[i.Index, feature][2]:
            del tags.at[i.Index, feature][2]
        if len(tags.at[i.Index, feature]) > 3 and 'Stayed' in tags.at[i.Index, feature][3]:
            del tags.at[i.Index, feature][3]
        if len(tags.at[i.Index, feature]) > 4 and 'Stayed' in tags.at[i.Index, feature][4]:
            del tags.at[i.Index, feature][4]
        if len(tags.at[i.Index, feature]) > 5 and 'Stayed' in tags.at[i.Index, feature][5]:
            del tags.at[i.Index, feature][5]

        if len(tags.at[i.Index, feature]) == 1:
            t1.append(tags.at[i.Index, feature][0])
            t2.append(0)
            t3.append(0)
            t4.append(0)
            t5.append(0)
        elif len(tags.at[i.Index, feature]) == 2:
            t1.append(tags.at[i.Index, feature][0])
            t2.append(tags.at[i.Index, feature][1])
            t3.append(0)
            t4.append(0)
            t5.append(0)
        elif len(tags.at[i.Index, feature]) == 3:
            t1.append(tags.at[i.Index, feature][0])
            t2.append(tags.at[i.Index, feature][1])
            t3.append(tags.at[i.Index, feature][2])
            t4.append(0)
            t5.append(0)
        elif len(tags.at[i.Index, feature]) == 4:
            t1.append(tags.at[i.Index, feature][0])
            t2.append(tags.at[i.Index, feature][1])
            t3.append(tags.at[i.Index, feature][2])
            t4.append(tags.at[i.Index, feature][3])
            t5.append(0)
        elif len(tags.at[i.Index, feature]) == 5:
            t1.append(tags.at[i.Index, feature][0])
            t2.append(tags.at[i.Index, feature][1])
            t3.append(tags.at[i.Index, feature][2])
            t4.append(tags.at[i.Index, feature][3])
            t5.append(tags.at[i.Index, feature][4])

    tags1 = pd.DataFrame(t1, columns=['T1'])
    tags2 = pd.DataFrame(t2, columns=['T2'])
    tags3 = pd.DataFrame(t3, columns=['T3'])
    tags4 = pd.DataFrame(t4, columns=['T4'])
    tags5 = pd.DataFrame(t5, columns=['T5'])


    data = pd.concat([data, tags1, tags2, tags3, tags4, tags5], axis=1)
    return data


# Encoding Features (which was not encoded anywhere)
def feature_encoder(x, cols):
    print("Encode Features...")
    for c in cols:
        try:
            lbl = LabelEncoder()
            lbl.fit(list(x[c].values))
            x[c] = lbl.transform(list(x[c].values))
        except:
            print('it stopes here', c)
    return x


# Scaling Features
def feature_scaling(x, a, b):
    print("Feature Scaling")
    normalized_x = np.zeros(x.shape)
    for i in range(x.shape[1]):
        normalized_x[:, i] = ((x[:, i]-min(x[:, i]))/(max(x[:, i])-min(x[:, i])))*(b-a)+a
    return normalized_x

def feature_scaling2(data, cols, a, b):
    print("Feature Scaling 2")
    data2 = pd.DataFrame(data)
    for c in cols:
        data2[c] = ((data[c] - min(data[c])) / (max(data[c]) - min(data[c]))) * (b - a) + a
    return data2

def fill_nulls(data):
    print("Fill Nulls...")
    for col in data:
        data[col] = data[col].fillna(data[col].mode()[0])
    return data

def encode_reviewer_score(data):
    print("Encode Reviewer Score...")
    data['Reviewer_Score'] = data['Reviewer_Score'].str.replace('High_Reviewer_Score', '0')
    data['Reviewer_Score'] = data['Reviewer_Score'].str.replace('Intermediate_Reviewer_Score', '1')
    data['Reviewer_Score'] = data['Reviewer_Score'].str.replace('Low_Reviewer_Score', '2')
    data['Reviewer_Score'] = data['Reviewer_Score'].astype(int)


def Corr(X, Y, X_test, Y_test):
    all_data_train = pd.concat([X, Y], axis=1)
    all_data_test = pd.concat([X_test, Y_test], axis=1)

    corr = all_data_train.corr()
    top_features = corr.index[abs(corr['Reviewer_Score']) > 0.2]

    # top_features corr plot
    plt.subplots(figsize=(5, 5))
    top_corr = all_data_train[top_features].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

    new_X_train = all_data_train[top_features]
    new_X_train = new_X_train.drop(columns=['Reviewer_Score'])
    linear_X_train = all_data_train['Average_Score']

    new_X_test = all_data_test[top_features]
    new_X_test = new_X_test.drop(columns=['Reviewer_Score'])
    linear_X_test = all_data_test['Average_Score']
    return new_X_train, new_X_test, linear_X_train, linear_X_test, top_features

