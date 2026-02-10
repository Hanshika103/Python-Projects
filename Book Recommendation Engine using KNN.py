import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# -------------------------------
# Filter users with >= 200 ratings
# -------------------------------
user_counts = ratings['user_id'].value_counts()
ratings = ratings[ratings['user_id'].isin(user_counts[user_counts >= 200].index)]

# -------------------------------
# Filter books with >= 100 ratings
# -------------------------------
book_counts = ratings['isbn'].value_counts()
ratings = ratings[ratings['isbn'].isin(book_counts[book_counts >= 100].index)]

# -------------------------------
# Merge with books to get titles
# -------------------------------
ratings = ratings.merge(books, on='isbn')

# -------------------------------
# Create book-user matrix
# -------------------------------
book_user_matrix = ratings.pivot_table(
    index='title',
    columns='user_id',
    values='rating'
).fillna(0)

# -------------------------------
# Convert to sparse matrix
# -------------------------------
book_user_sparse = csr_matrix(book_user_matrix.values)

# -------------------------------
# Train KNN model
# -------------------------------
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(book_user_sparse)

# -------------------------------
# Recommendation function
# -------------------------------
def get_recommends(book_title):
    if book_title not in book_user_matrix.index:
        return "Book not found"

    book_index = book_user_matrix.index.get_loc(book_title)

    distances, indices = model.kneighbors(
        book_user_sparse[book_index],
        n_neighbors=6
    )

    recommends = []
    for i in range(1, len(distances[0])):
        recommends.append([
            book_user_matrix.index[indices[0][i]],
            distances[0][i]
        ])

    return [book_title, recommends]
