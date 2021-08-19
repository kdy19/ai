import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/rickiepark/handson-ml2/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == '__main__':
    # fetch_housing_data()
    housing = load_housing_data()
    # print(housing.head())
    # print(housing.info())
    # print(housing.describe())
    # housing.hist(bins=50, figsize=(20,15))
    # plt.show()

    train_set, test_set = split_train_test(housing, 0.2)
    print(len(train_set))
    print(len(test_set))

    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    # housing['income_cat'].hist()
    # plt.show()

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))

    # housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
    # plt.show()

    '''
    housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                s=housing['population']/100, label='population', figsize=(10,7),
                c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True,
                sharex=False
    )
    plt.show()
    '''

    # corr_matrix = housing.corr()
    # print(corr_matrix['median_house_value'].sort_values(ascending=False))

    '''
    housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
    housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
    housing['population_per_household'] = housing['population'] / housing['households']

    corr_matrix = housing.corr()
    print(corr_matrix['median_house_value'].sort_values(ascending=False))
    '''

    housing_cat = housing[['ocean_proximity']]
    print(housing_cat.head(10))

    ordinal_encoder = OrdinalEncoder()
    housing_cat_encode = ordinal_encoder.fit_transform(housing_cat)
    print(housing_cat_encode[:10])
    print(ordinal_encoder.categories_)
