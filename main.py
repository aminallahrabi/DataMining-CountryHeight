# //////    packages   /////////
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# /////////////     read data        ///////////////////

df = pd.read_csv('Country_Heights.csv')

# print(df.shape)
# print(df.info)
# print(len(df))

# //////////   Q1   /////////////////
def showOnVector():
    xy = df.rename(columns={'Country':'C','Sex':'S','Year':'Y','Age group':'A','Mean height':'H'})
    h = xy.max()
    fig,ax = plt.subplots()
    ax.bar(xy.columns.values,np.array(h))
    ax.set_title('Q1')
    ax.set_xlabel('Columns name')
    ax.set_ylabel('Count')
    plt.show()
# //////////  Q2 //////////////////
def getRows():
    print(len(df.index))

# //////////  Q3 //////////////////
def BoysMeanHeight():
    BMH = df[df['Sex'] == 1]['Mean height'].mean()
    print(BMH)

# //////////  Q4 //////////////////
def Girls19MeanHeight():
    GMH = df[(df['Sex'] == 0) & (df['Age group'] == 19)]['Mean height'].mean()
    print(GMH)

# //////////  Q5 //////////////////
def heighestBoys16Country():
    HBC = df[(df['Sex'] == 1) & (df['Age group'] == 16)].sort_values(by=['Mean height'], ascending=False)['Country'].iloc[0]
    print(HBC)

# //////////  Q6 //////////////////
def iranianInformation():
    Iin = df[df['Country'] == 83]
    matrix = Iin.to_numpy()
    print(matrix)

# //////////  Q7-1 //////////////////
def meanHeightUpper167():
    xy = df[(df['Sex'] == 1) & (df['Mean height'] > 167)].sort_values(by=['Age group'], ascending=True)['Age group']
    print(xy.iloc[0])

# //////////  Q7-2 //////////////////
def meanHeightUpper155():
    xy = df[(df['Sex'] == 0) & (df['Mean height'] > 155)].sort_values(by=['Age group'], ascending=True)['Age group']
    print(xy.iloc[0])

# /////////   Q8-1   /////////////////////
def threeTopCountry():
    xy = df[(df['Sex'] == 0)].sort_values(by=['Mean height'], ascending=False).drop_duplicates(subset=['Country'])[
        'Country']
    print(xy.iloc[0:3].values)

# //////// Q8-2   ///////////////////////
def difMinBoysGirls19():
    xyb = df[(df['Sex'] == 1)].sort_values(by=['Mean height'], ascending=True)['Mean height']
    xyg = df[(df['Sex'] == 0)].sort_values(by=['Mean height'], ascending=True)['Mean height']
    dif = xyg.iloc[0:1].values - xyb.iloc[0:1].values
    print(f' MIN boys - MIN girls = {dif}')

# /////////  Q9   ////////////////////
def meanHeaightAgegroups():
    xyb = df[df['Sex'] == 1].groupby(['Age group'])['Mean height'].mean()
    xyg = df[df['Sex'] == 0].groupby(['Age group'])['Mean height'].mean()
    print(f'Mean height BOYS by {xyb}', '\n', f'Mean height GIRLS by {xyg}')

def chartHeightPerAgeBoys():
    age = np.array(df[df['Sex'] == 1]['Age group'])
    height = np.array(df[df['Sex'] == 1]['Mean height'])
    fig,ax = plt.subplots()
    ax.bar(age,height)
    ax.set_title('Q9')
    ax.set_xlabel('Age groups')
    ax.set_ylabel('Height')
    plt.savefig('HeightPerAgeBoys.png')
    plt.show()

def chartHeightPerAgeGirls():
    age = np.array(df[df['Sex'] == 0]['Age group'])
    height = np.array(df[df['Sex'] == 0]['Mean height'])
    fig,ax = plt.subplots()
    ax.bar(age,height)
    ax.set_title('Q9')
    ax.set_xlabel('Age groups')
    ax.set_ylabel('Height')
    plt.savefig('HeightPerAgeBoys.png')
    plt.show()

# ////////  Q10   ///////////////
def heightPerSex19():
    height = df[df['Age group'] == 19]['Mean height']
    sex = df[df['Age group'] == 19]['Sex']
    plt.scatter(sex, height, marker='*', color='r')
    plt.show()

def equalDomainBoysGirls():
    boys = df[(df['Age group'] == 19) & (df['Sex'] == 1)]['Mean height']
    girls = df[(df['Age group'] == 19) & (df['Sex'] == 0)]['Mean height']
    MAX = 0
    MIN = 0
    if boys.max() > girls.max():
        MAX = girls.max()
    else:
        MAX = boys.max()
    if boys.min() < girls.min():
        MIN = girls.min()
    else:
        MIN = boys.min()

    CountBoys = len(
        df[(df['Age group'] == 19) & (df['Sex'] == 1) & (df['Mean height'] <= MAX) & (df['Mean height'] >= MIN)])
    CountGirls = len(
        df[(df['Age group'] == 19) & (df['Sex'] == 0) & (df['Mean height'] <= MAX) & (df['Mean height'] >= MIN)])
    print(f'Count Boys ={CountBoys}')
    print(f'Count Girls ={CountGirls}')

# ////////   Q11   ////////////////
def normalize():
    mean = df[(df['Sex'] == 1) & (df['Age group'] == 13) & (df['Country'] == 83)]['Mean height'].mean()
    std = df[(df['Sex'] == 1) & (df['Age group'] == 13) & (df['Country'] == 83)]['Mean height'].std()
    norm = ((df[(df['Sex'] == 1) & (df['Age group'] == 13) & (df['Country'] == 83)]['Mean height'] - mean) / std)
    print(norm)

# ///////////     RUN FUNCTIONS    ///////////////////

#q1
# showOnVector()
#q2
# getRows()
#q3
# BoysMeanHeight()
#q4
# Girls19MeanHeight()
#q5
# heighestBoys16Country()
#q6
# iranianInformation()
#q7-1
# meanHeightUpper167()
#q7-2
# meanHeightUpper155()
#q8-1
# threeTopCountry()
#q8-2
# difMinBoysGirls19()
#q9
# meanHeaightAgegroups()
# chartHeightPerAgeBoys()
# chartHeightPerAgeGirls()
#q10
# heightPerSex19()
# equalDomainBoysGirls()
#q11
# normalize()
