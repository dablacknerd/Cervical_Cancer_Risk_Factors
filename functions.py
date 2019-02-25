import random
import pandas as pd
import numpy as np

def shuffle_dataframe(dataframe):
    shuffled_index = [idx for idx in dataframe.index]
    random.shuffle(shuffled_index)
    temp_dataframe = dataframe.loc[shuffled_index].copy()
    return pd.DataFrame(temp_dataframe.values,columns=temp_dataframe.columns)

def split_dataframe(dataframe,cv_size):
    dataframes = []
    running_number = dataframe.shape[0]/cv_size
    counter = 1
    start = 0
    end = int(running_number)
    while counter <= cv_size:
        #print(counter)
        #print(start)
        #print(end)
        #print('-----------------------------------')
        dataframes.append(dataframe.iloc[start:end])
        counter = counter + 1
        start = start + int(running_number)
        end = end + int(running_number)
    dataframes.append(dataframe.iloc[end:])
    return dataframes

def multilabel_accuracy(true_label,predicted_label):
    acc_count = 0
    for i in range(0,len(true_label)):
        t = ''.join(true_label[i])
        p = ''.join(predicted_label[i])
        if t == p:
            acc_count = acc_count + 1
        else:
            continue
    return round(float(acc_count)/float(len(true_label)),2)

def create_multilabel_target(df, targets):
    #print(df)
    y1 = df[targets].copy()
    y_multilabel =[]
    for idx in y1.index:
        #print(idx)
        #row =[str(y1.iloc[idx][0]),str(y1.iloc[idx][1]),str(y1.iloc[idx][2]),str(y1.iloc[idx][3])]
        row = [str(r) for r in y1.loc[idx].tolist()]
        y_multilabel.append(row)
    return y_multilabel

def fit_model(model,features,targets,data):
    X = data[features]
    y = create_multilabel_target(data, targets)
    model.fit(X,y)
    return model

def transform_preds(predictions):
    xrows =[]
    for row in predictions:
        nrow = [str(r) for r in row]
        xrows.append(nrow)
    return xrows

def cross_validation(df,targets,model,cv=5):
    accuracy_list = []
    counter = 0
    frames = []
    features = list(set(df.columns) - set(targets))
    while counter < cv:
        #print("cv {}".format(counter))
        shuffled_frame = shuffle_dataframe(df)
        split_frames = split_dataframe(shuffled_frame,cv)
        k = random.randint(0,cv-1)
        #print("K is {}".format(k))
        for i in range(0,len(split_frames)):
            if i == k:
                continue
            else:
                frames.append(split_frames[i])
        #counter = counter + 1
        validation = split_frames[k]
        #print(validation)
        training = pd.concat(frames)
        frames =[]
        model2 = fit_model(model,features,targets,training)
        #print(x_valid)
        x_valid = validation[features]
        y_valid = create_multilabel_target(validation, targets)
        #print(x_valid.shape)
        model2_preds = model2.predict(x_valid)
        t_model2_preds = transform_preds(model2_preds)
        accuracy = multilabel_accuracy(y_valid,t_model2_preds)
        accuracy_list.append(accuracy)
        counter = counter + 1
        #print(accuracy)
        #print('----------')
    return accuracy_list

def average_cv_accuracy(results):
    return round(np.array(results).mean(),2)