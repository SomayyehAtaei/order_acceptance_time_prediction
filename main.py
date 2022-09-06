import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn import preprocessing
from sklearn.svm import SVR
import sklearn.cluster as sc
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

pause_plot_time = 100
def clustering_lat_long(train_X, test_X):
    points_1 = pd.concat((train_X['source_latitude'], train_X['source_longitude']), axis=1)
    points_1.rename(columns={'source_latitude': 'latitude', 'source_longitude': 'longitude'}, inplace=True)

    points_2 = pd.concat((train_X['destination_latitude'], train_X['destination_longitude']), axis=1)
    points_2.rename(columns={'destination_latitude': 'latitude', 'destination_longitude': 'longitude'}, inplace=True)
    points = pd.concat((points_1, points_2), axis=0)

    km_model = sc.KMeans()
    param_grid = {"n_clusters": range(7, 20)}
    search = GridSearchCV(km_model, param_grid=param_grid)
    km_model = search.fit(points)
    train_X['source_cluster'] = km_model.predict(points_1)
    train_X['destination_cluster'] = km_model.predict(points_2)

    points_1 = pd.concat((test_X['source_latitude'], test_X['source_longitude']), axis=1)
    points_1.rename(columns={'source_latitude': 'latitude', 'source_longitude': 'longitude'}, inplace=True)

    points_2 = pd.concat((test_X['destination_latitude'], test_X['destination_longitude']), axis=1)
    points_2.rename(columns={'destination_latitude': 'latitude', 'destination_longitude': 'longitude'}, inplace=True)

    test_X['source_cluster'] = km_model.predict(points_1)
    test_X['destination_cluster'] = km_model.predict(points_2)

    return train_X, test_X

def main():
##########################################################################################################
# Load data and calculate it's statistics
##########################################################################################################
    # data = pd.read_excel('./data/Data_task 1.xlsx')
    # print('Dataframe dimensions:', data.shape)
    # data.to_pickle('./data/data_task1.pkl')
    data = pd.read_pickle('./data/data_task1.pkl')
    print('data dimensions:', data.shape)
    print('###########################################################################################')

    data['first_created_at'] = pd.to_datetime(data['first_created_at'])
    data['weekday'] = data['first_created_at'].dt.day_name()

    data_profile = pp.ProfileReport(data)
    data_profile.to_file("Data_Statistics.html")
    print('Data Statistics are ready in Data_Statistics.html file...')
    print('###########################################################################################')
    print("Skewness: %f" % data['create_to_acc'].skew())
    print("Kurtosis: %f" % data['create_to_acc'].kurt())
    print('###########################################################################################')

#histogram
    sns.distplot(data['create_to_acc'])
    plt.show(block=False)
    plt.pause(pause_plot_time)
    plt.close()

##########################################################################################################
# Removing records with missing values
##########################################################################################################
    data = data[data['final_status'] == 'DELIVERED']

    print('Delete All CANCELED Orders:')

    data_null_percent = pd.DataFrame(data.dtypes).T.rename(index={0: 'column type'})
    data_null_percent = data_null_percent.append(
        pd.DataFrame(data.isnull().sum()).T.rename(index={0: 'number_of_null_values'}))
    data_null_percent = data_null_percent.append(
        pd.DataFrame(data.isnull().sum() / data.shape[0] * 100).T.rename(index={0: 'percent_of_null_values'}))
    data_null_percent = data_null_percent.transpose()
    data_null_percent = data_null_percent[data_null_percent['percent_of_null_values'] > 0]
    print(data_null_percent)
    print('###########################################################################################')

    data.dropna(axis=0, subset=['delivered_at'], inplace=True)
    print('data dimensions after delete CANCELED and delivered_at=null records:', data.shape)
    print('###########################################################################################')

    data = data.drop(['date', 'first_created_at', 'accepted_at', 'delivered_at'], axis=1)
##########################################################################################################
# Univarient Outlier Detection
##########################################################################################################
    data = data[data['create_to_acc'] < 22]

    print("Skewness: %f" % data['create_to_acc'].skew())
    print("Kurtosis: %f" % data['create_to_acc'].kurt())


    print('data dimensions after univarient outlier removal:', data.shape)
    print('###########################################################################################')

#histogram
    sns.distplot(data['create_to_acc'])
    plt.show(block=False)
    plt.pause(pause_plot_time)
    plt.close()

#scatterplot
    sns.set()
    cols = ['create_to_acc', 'total_distance', 'final_customer_fare', 'hour', 'weekday']
    sns.pairplot(data[cols], size = 2.5)
    plt.show(block=False)
    plt.pause(pause_plot_time)
    plt.close()
##########################################################################################################
    data_X = data.drop(['create_to_acc'], axis=1)
    data_y = data['create_to_acc']

    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.3)
    train = pd.concat((train_X, train_y), axis=1)

    print('train dimensions:', train.shape)
    print('###########################################################################################')

#bivariate Outliers
    train1 = pd.concat([train['create_to_acc'], train['total_distance']], axis=1)
    train1.plot.scatter(x='total_distance', y='create_to_acc')
    plt.show(block=False)
    plt.pause(pause_plot_time)
    plt.close()

    train = train[train['total_distance'] < 10]

    train1 = pd.concat([train['create_to_acc'], train['final_customer_fare']], axis=1)
    train1.plot.scatter(x='final_customer_fare', y='create_to_acc')
    plt.show(block=False)
    plt.pause(pause_plot_time)
    plt.close()

    train = train[train['final_customer_fare'] < 300000]

    train1 = pd.concat([train['create_to_acc'], train['hour']], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x="hour", y="create_to_acc", data=train1)
    plt.show(block=False)
    plt.pause(pause_plot_time)
    plt.close()

    train1 = pd.concat([train['create_to_acc'], train['weekday']], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x="weekday", y="create_to_acc", data=train1)
    plt.show(block=False)
    plt.pause(pause_plot_time)
    plt.close()

    sns.set()
    cols = ['create_to_acc', 'total_distance', 'final_customer_fare', 'hour', 'weekday']
    sns.pairplot(train[cols], size=2.5)
    plt.show(block=False)
    plt.pause(pause_plot_time)
    plt.close()

##########################################################################################################
# Clustering source and destination based on Latitude/Longitude
##########################################################################################################
    train_X = train.drop(['create_to_acc'], axis=1)
    train_X, test_X = clustering_lat_long(train_X, test_X)

    plt.subplots(figsize=(8, 6))
    plt.scatter(train_X[:]['source_latitude'], train_X[:]['source_longitude'], c=train_X[:]['source_cluster'], s=50, cmap='inferno')

    plt.subplots(figsize=(8, 6))
    plt.scatter(train_X[:]['destination_latitude'], train_X[:]['destination_longitude'], c=train_X[:]['destination_cluster'], s=50,  cmap='inferno')
    plt.show(block=False)
    plt.pause(pause_plot_time)
    plt.close()

    train_X = train_X[['hour', 'total_distance', 'final_customer_fare', 'source_cluster', 'destination_cluster']]
    train_y = train['create_to_acc']

    test_X = test_X[['hour', 'total_distance', 'final_customer_fare', 'source_cluster', 'destination_cluster']]


    print('train dimensions after removing outliers and less important columns:', train.shape)
    print('###########################################################################################')

##########################################################################################################
# Normalization
##########################################################################################################
    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

##########################################################################################################
# Training ML Models
##########################################################################################################
    lr_model = LinearRegression()
    lr_model = lr_model.fit(train_X, train_y)
    score = lr_model.score(test_X, test_y)
    test_y_pred = lr_model.predict(test_X)
    res = test_y - test_y_pred
    on_time = (res[res <= 5].shape[0] / test_y.shape[0]) * 100
    more_than_five_min_late = (res[(res > 5) & (res <= 10)].shape[0] / test_y.shape[0]) * 100
    more_than_ten_min_late = (res[res > 10].shape[0] / test_y.shape[0]) * 100
    from sklearn import metrics
    print('Linear Regresion Results:')
    print('Score: {:.2f}'.format(score))
    print('Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(test_y, test_y_pred)))
    print('Mean Square Error: {:.2f}'.format(metrics.mean_squared_error(test_y, test_y_pred)))
    print('On time: {:.2f} %'.format(on_time))
    print('More than 5 mins late: {:.2f} %'.format(more_than_five_min_late))
    print('More than 10 mins late: {:.2f} %'.format(more_than_ten_min_late))

    print('=============================================================================')
    rf_model = RandomForestRegressor()
    rf_model = rf_model.fit(train_X, train_y)
    score = rf_model.score(test_X, test_y)
    test_y_pred = rf_model.predict(test_X)
    res = test_y - test_y_pred
    on_time = (res[res <= 5].shape[0] / test_y.shape[0]) * 100
    more_than_five_min_late = (res[(res > 5) & (res <= 10)].shape[0] / test_y.shape[0]) * 100
    more_than_ten_min_late = (res[res > 10].shape[0] / test_y.shape[0]) * 100

    print('Random Forest Regression Results:')
    print('Score: {:.2f}'.format(score))
    print('Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(test_y, test_y_pred)))
    print('Mean Square Error: {:.2f}'.format(metrics.mean_squared_error(test_y, test_y_pred)))
    print('On time: {:.2f} %'.format(on_time))
    print('More than 5 mins late: {:.2f} %'.format(more_than_five_min_late))
    print('More than 10 mins late: {:.2f} %'.format(more_than_ten_min_late))


    print('=============================================================================')
    gb_model = GradientBoostingRegressor(random_state=1)
    gb_model = gb_model.fit(train_X, train_y)
    score = gb_model.score(test_X, test_y)
    test_y_pred = gb_model.predict(test_X)
    res = test_y - test_y_pred
    on_time = (res[res <= 5].shape[0] / test_y.shape[0]) * 100
    more_than_five_min_late = (res[(res > 5) & (res <= 10)].shape[0] / test_y.shape[0]) * 100
    more_than_ten_min_late = (res[res > 10].shape[0] / test_y.shape[0]) * 100

    print('Gradient Boosting Regression Results:')
    print('Score: {:.2f}'.format(score))
    print('Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(test_y, test_y_pred)))
    print('Mean Square Error: {:.2f}'.format(metrics.mean_squared_error(test_y, test_y_pred)))
    print('On time: {:.2f} %'.format(on_time))
    print('More than 5 mins late: {:.2f} %'.format(more_than_five_min_late))
    print('More than 10 mins late: {:.2f} %'.format(more_than_ten_min_late))

    print('=============================================================================')
    vr_model = VotingRegressor([('lr', lr_model), ('rf', rf_model), ('gb', gb_model)])
    vr_model = vr_model.fit(train_X, train_y)
    score = vr_model.score(test_X, test_y)
    test_y_pred = vr_model.predict(test_X)
    res = test_y - test_y_pred
    on_time = (res[res <= 5].shape[0] / test_y.shape[0]) * 100
    more_than_five_min_late = (res[(res > 5) & (res <= 10)].shape[0] / test_y.shape[0]) * 100
    more_than_ten_min_late = (res[res > 10].shape[0] / test_y.shape[0]) * 100

    print('Voting Regression Results:')
    print('Score: {:.2f}'.format(score))
    print('Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(test_y, test_y_pred)))
    print('Mean Square Error: {:.2f}'.format(metrics.mean_squared_error(test_y, test_y_pred)))
    print('On time: {:.2f} %'.format(on_time))
    print('More than 5 mins late: {:.2f} %'.format(more_than_five_min_late))
    print('More than 10 mins late: {:.2f} %'.format(more_than_ten_min_late))

    print('=============================================================================')
    sv_model = SVR(C=1.0, epsilon=0.2)
    sv_model = sv_model.fit(train_X, train_y)
    score = sv_model.score(test_X, test_y)
    test_y_pred = sv_model.predict(test_X)
    res = test_y - test_y_pred
    on_time = (res[res <= 5].shape[0] / test_y.shape[0]) * 100
    more_than_five_min_late = (res[(res > 5) & (res <= 10)].shape[0] / test_y.shape[0]) * 100
    more_than_ten_min_late = (res[res > 10].shape[0] / test_y.shape[0]) * 100

    print('SVM Regression Results:')
    print('Score: {:.2f}'.format(score))
    print('Mean Absolute Error: {:.2f}'.format(metrics.mean_absolute_error(test_y, test_y_pred)))
    print('Mean Square Error: {:.2f}'.format(metrics.mean_squared_error(test_y, test_y_pred)))
    print('On time: {:.2f} %'.format(on_time))
    print('More than 5 mins late: {:.2f} %'.format(more_than_five_min_late))
    print('More than 10 mins late: {:.2f} %'.format(more_than_ten_min_late))

if __name__ == '__main__':
    main()
