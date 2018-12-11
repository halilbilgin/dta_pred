import pandas as pd
from sklearn.utils import resample

def under_sampling(X_train, y_train):
    df_all = pd.concat((X_train, pd.DataFrame({'value': y_train}, index=y_train.index)), axis=1)

    df_majority = df_all[df_all.value == 0]
    df_minority = df_all[df_all.value == 1]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=df_minority.shape[0],  # to match minority class
                                       random_state=123)
    # Join together class 0's target vector with the downsampled class 1's target vector
    df_downsampled = pd.concat([df_majority_downsampled, df_minority], axis=0)
    y_downsampled = df_downsampled.value
    X_downsampled = df_downsampled.drop('value', axis=1)
    return X_downsampled, y_downsampled

def over_sampling(X_train, y_train):
    df_all = pd.concat((X_train, pd.DataFrame({'value': y_train}, index=y_train.index)), axis=1)

    df_majority = df_all[df_all.value == 0]
    df_minority = df_all[df_all.value == 1]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=df_majority.shape[0],  # to match majority class
                                     random_state=123)  # reproducible results
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled], axis=0)
    y_upsampled = df_upsampled.value
    X_upsampled = df_upsampled.drop('value', axis=1)

    return X_upsampled, y_upsampled