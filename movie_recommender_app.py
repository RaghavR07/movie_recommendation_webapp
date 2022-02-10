import numpy as np
import pandas as pd

see = pd.read_csv(r"C:\Users\ASUS\Desktop\Movie Recommend Project\Model Training\tmdb_5000_credits.csv")
see.to_csv("credits.csv")
credits=pd.read_csv("credits.csv")

see = pd.read_csv(r"C:\Users\ASUS\Desktop\Movie Recommend Project\Model Training\tmdb_5000_movies.csv")
see.to_csv("movies.csv")
movies=pd.read_csv("movies.csv")

credits=credits[["movie_id","title","cast","crew"]]

# print(credits.shape)
# print(movies.shape)

movies=movies.merge(credits,on='title')
# print(movies.shape)

# Selecting useful Columns for our model

movies = movies[['movie_id','title','overview','genres','keywords','cast']]

# Creating Tag Column from existing Columns

# print(movies.isnull().sum())

movies.dropna(inplace=True)

# print(movies.iloc[0].genres)

#Converts String of List to List
import ast
#ast.literal_eval()

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
# print(movies['genres'])

movies['keywords'] = movies['keywords'].apply(convert)
# print(movies['keywords'])


# Fetching Top 3 cast from each movie for our model

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L

movies['cast'] = movies['cast'].apply(convert3)
(movies['cast'])


movies['overview']=movies['overview'].apply(lambda x:x.split())
# print(movies['overview'])

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast']


new = movies.drop(columns=['overview','genres','keywords','cast'])

new['tags'] = new['tags'].apply(lambda x: " ".join(x))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(new['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)

# new[new['title'] == 'The Lego Movie'].index[0]

def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

print(recommend('Gandhi'))


