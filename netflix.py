import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD, evaluate
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

if __name__ == "__main__":

    # ---------Reading data---------
    df = pd.read_csv('./combined_data_1.txt', header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])

    df['Rating'] = df['Rating'].astype(float)

    print('Dataset 1 shape: {}'.format(df.shape))
    print('-Dataset examples-')
    print(df.iloc[::5000000, :])

    df.index = np.arange(0, len(df))
    print('Full dataset shape: {}'.format(df.shape))
    print('-Dataset examples-')
    print(df.iloc[::5000000, :])

    # ---------Data cleaning---------
    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan['Rating'] == True]
    df_nan = df_nan.reset_index()

    movie_np = []
    movie_id = 1

    for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
        temp = np.full((1, i - j - 1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    # Account for last record and corresponding length
    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    movie_np = np.append(movie_np, last_record)

    print('Movie numpy: {}'.format(movie_np))
    print('Length: {}'.format(len(movie_np)))

    df = df[pd.notnull(df['Rating'])]

    df['Movie_Id'] = movie_np.astype(int)
    df['Cust_Id'] = df['Cust_Id'].astype(int)
    print('-Dataset examples-')
    print(df.iloc[::5000000, :])

    # ---------DataSet optimization---------
    f = ['count', 'mean']

    df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.8), 0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

    print('Movie minimum times of review: {}'.format(movie_benchmark))

    df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
    df_cust_summary.index = df_cust_summary.index.map(int)
    cust_benchmark = round(df_cust_summary['count'].quantile(0.8), 0)
    drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

    print('Customer minimum times of review: {}'.format(cust_benchmark))
    print('Original Shape: {}'.format(df.shape))
    df = df[~df['Movie_Id'].isin(drop_movie_list)]
    df = df[~df['Cust_Id'].isin(drop_cust_list)]
    print('After Trim Shape: {}'.format(df.shape))
    print('-Data Examples-')
    print(df.iloc[::5000000, :])
    df_p = pd.pivot_table(df, values='Rating', index='Cust_Id', columns='Movie_Id')

    print(df_p.shape)

    # ---------Data Mapping---------
    df_title = pd.read_csv('./movie_titles.csv', encoding="ISO-8859-1", header=None,
                           names=['Movie_Id', 'Year', 'Name'])
    df_title.set_index('Movie_Id', inplace=True)
    print (df_title.head(10))

    # ---------Recommendation system impl---------
    reader = Reader()

    # get just top 100K rows for faster run time
    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:100000], reader)
    data.split(n_folds=3)

    svd = SVD()
    evaluate(svd, data, measures=['RMSE', 'MAE'])

    # Testing for user ID 7
    df_7 = df[(df['Cust_Id'] == 7) & (df['Rating'] == 5)]
    df_7 = df_7.set_index('Movie_Id')
    df_7 = df_7.join(df_title)['Name']
    print(df_7)  # Previous movies

    user_7 = df_title.copy()
    user_7 = user_7.reset_index()
    user_7 = user_7[~user_7['Movie_Id'].isin(drop_movie_list)]

    # getting full dataset
    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)

    trainset = data.build_full_trainset()
    svd.train(trainset)

    user_7['Estimate_Score'] = user_7['Movie_Id'].apply(lambda x: svd.predict(7, x).est)

    user_7 = user_7.drop('Movie_Id', axis=1)

    user_7 = user_7.sort_values('Estimate_Score', ascending=False)
    # Recommended movies Finally!!!!
    print(user_7.head(10))
