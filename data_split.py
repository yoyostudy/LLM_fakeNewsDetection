import pandas as pd

def split_train_test_val(dataset, val_ratio):
    val_set = dataset.head(int(len(dataset)*val_ratio))
    train_test_set = dataset.tail(len(dataset) - len(val_set))
    return val_set, train_test_set

training_data = pd.read_csv('data/fake_news_raw.csv', index_col=0).dropna()
training_data = training_data.sample(frac=1).reset_index(drop=True)
val_set, train_test_set = split_train_test_val(training_data, 0.2)
val_set.to_csv('data/fake_news_validation_set.csv', index=False)
train_test_set.to_csv('data/fake_news_training_set.csv', index=False)
