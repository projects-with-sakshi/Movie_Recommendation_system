
import numpy as np
import pandas as pd

movies = pd.read_csv(r"/content/tmdb_5000_movies.csv")
credits = pd.read_csv(r"/content/tmdb_5000_credits.csv")

movies.head(1)

credits.head(1)['cast'].values

movies.shape

credits.shape

movies = movies.merge(credits,on='title')

movies.head(1)

movies.info()

# Important columns for recommendor system
# genres , id , keywords , title,overview,cast,crew

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.head()

movies.dropna(inplace=True)

movies.isnull().sum()

movies.duplicated().sum()

movies.iloc[0].genres

# ['Action','Adventure','Fantasy','SciFi']

import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

def convert(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

movies['genres'] = movies['genres'].apply(convert)

movies['genres']

movies.head()

movies['keywords'] = movies['keywords'].apply(convert)

movies.head()

movies['cast'][0]

def convert3(obj):
  L=[]
  counter=0
  for i in ast.literal_eval(obj):
    if counter !=3:
      L.append(i['name'])
      counter +=1
    else:
      break
  return L

movies['cast'] = movies['cast'].apply(convert3)

movies.head()

movies['crew'][0]

def fetch_director(obj):
  L=[]
  for i in ast.literal_eval(obj):
    if i['job'] == 'Director':
      L.append(i['name'])
      break
  return L

movies['crew']= movies['crew'].apply(fetch_director)

movies.head()

movies['overview'][0]

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies.head()

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head()

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies.head()

new_df = movies[['movie_id','title','tags']]

new_df

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

new_df.head()

new_df['tags'][0]

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

new_df.head()

new_df['tags'][0]

new_df['tags'][1]

# textvectorisation >> process of converting text into vectors

!pip install nltk

import nltk  ##natural language processing

# steming >> process of converting similar words into one word

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
  y=[]

  for i in text.split():
    y.append(ps.stem(i))

  return " ".join(y)

stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')

new_df['tags'] = new_df['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
 cv = CountVectorizer(max_features=5000,stop_words='english')

cv.fit_transform(new_df['tags']).toarray().shape

vectors = cv.fit_transform(new_df['tags']).toarray()

vectors[0]

cv.get_feature_names_out()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

cosine_similarity(vectors).shape

similarity[0].shape

similarity[1]

new_df[new_df['title'] == 'Batman Begins'].index[0]

sorted(similarity[0],reverse = True)

sorted(list(enumerate(similarity[0])),reverse = True,key=lambda x:x[1])[1:6]

def recommend(movie):
  movie_index = new_df[new_df['title'] == movie].index[0]
  distances = similarity[movie_index]
  movie_list=sorted(list(enumerate(distances)),reverse = True,key=lambda x:x[1])[1:6]

  for i in movie_list:
    print(new_df.iloc[i[0]].title)

new_df.iloc[1216].title

recommend('Avatar')

recommend('Batman Begins')

new_df.to_dict()

import pickle
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))

