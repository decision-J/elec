def MLP():
    BATCH_SIZE = 24

    import tensorflow as tf
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense, Activation

    import numpy as np
    import pandas as pd

    from sklearn import preprocessing
    from collections import deque

    import itertools

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(rc={'figure.figsize':(20, 8)})

    import random
    import time

    # Random seed 고정
    # tf.reset_default_graph()
    # seed = 1
    # np.random.seed(seed)
    # tf.set_random_seed(seed)
    # random.seed(seed)

    #데이터셋 불러오기
    filepath = 'C:/Users/JYW/Desktop/Github/repository/elec/Machine_learning_dataset_191003_v3_2.csv'

    #데이터셋을 pandas를 통해서 읽기
    dt17 = pd.read_csv(filepath, index_col = 0)
    #데이터셋 copy 및 앞 부분 보기
    pm3 = dt17.copy()

    #데이터 전처리 작업 및 사용할 변수 재정리
    scaler = preprocessing.MinMaxScaler()
    cols = ['year','month', 'date', 'hour', 'day', 'temp_tt','rainfall_tt','ss_all','elec']
    pm3 = pm3[cols]

    #Data split index 설정
    pm3 = pm3.reset_index(drop=True)

    pm3["month"] = list(['{0:02d}'.format(pm3.month[i]) for i in range(0, len(pm3))])
    pm3["date"] = list(['{0:02d}'.format(pm3.date[i]) for i in range(0, len(pm3))])

    pm3["time"] = (pm3.year.astype(str) + '' + pm3.month.astype(str) + '' + pm3.date.astype(str)).astype(int)

    train_index = pm3[20100101 <= pm3["time"]]
    train_index = train_index[train_index["time"] <= 20171221]
    train_index = [train_index.index[0], train_index.index[len(train_index)-1]]

    valid_index = pm3[20171222 <= pm3["time"]]
    valid_index = valid_index[valid_index["time"] <= 20171228]
    valid_index = [valid_index.index[0], valid_index.index[len(valid_index)-1]]

    test_index = pm3[20171229 <= pm3["time"]]
    test_index = test_index[test_index["time"] <= 20180105]
    test_index = [test_index.index[0], test_index.index[len(test_index)-1]]

    #더미변수 생성
    pm3 = pd.get_dummies(pm3, columns = ['month', 'date', 'hour', 'day'])

    #전력부하 scale 조정
    pm3['elec'] = pm3['elec']/100000

    #기온 시차항 생성
    pm3['shift_temp_1'] = pm3['temp_tt'].shift(1)

    #기타 변수들 scaler인 MinMax 사용하여 조정
    pm3['year'] = scaler.fit_transform(np.array(pm3['year']).reshape(-1,1))
    pm3['temp_tt'] = scaler.fit_transform(np.array(pm3['temp_tt']).reshape(-1,1))
    pm3['shift_temp_1'] = scaler.fit_transform(np.array(pm3['shift_temp_1']).reshape(-1,1))
    pm3['rainfall_tt'] = scaler.fit_transform(np.array(pm3['rainfall_tt']).reshape(-1,1))

    #기온제곱항 만들기
    pm3['temp_tt_square'] = pm3['temp_tt']**2
    #최종적으로 사용할 변수
    re_cols = ['temp_tt','temp_tt_square','shift_temp_1','year','month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06', 'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12','date_01', 'date_02', 'date_03', 'date_04', 'date_05', 'date_06', 'date_07', 'date_08', 'date_09', 'date_10', 'date_11', 'date_12','date_13', 'date_14', 'date_15', 'date_16', 'date_17', 'date_18', 'date_19', 'date_20', 'date_21', 'date_22', 'date_23', 'date_24','date_25', 'date_26', 'date_27','date_28','date_29','date_30','date_31','day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7','hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12','hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'ss_all','elec']

    #변수의 개수 보기
    pm3 = pm3[re_cols]
    pm3.dropna(inplace = True)

    #학습/검증/평가 기간 확인해보기
    training = pm3.iloc[train_index[0]:train_index[1],:]
    validation = pm3.iloc[valid_index[0]:valid_index[1],:]
    test = pm3.iloc[test_index[0]:test_index[1],:]

    #평가기간 iteration을 위해서 사전에 점검
    for i in range(53):
            week = i+0
            interval = 168
            n = valid_index[1]
            m = test_index[1]
            test_week = pm3.iloc[n+interval*week:m+interval*week,:]
            test_XX = test_week
            # print(test_XX)

    #엑셀 파일로 저장하기 위해 openpyxl 사용
    from openpyxl import Workbook
    wb = Workbook()
    weeklyload_18 = wb.active
    weeklyload_18.title = 'none'
    wb.save('C:/Users/JYW/Desktop/Github/repository/elec/weeklyload_18_model_A_12_adjusted_sample_period_for_explanation.xlsx')

    #2018년 기간 동안 1주(8일)씩 예측치를 도출하도록 진행
    #각 주별로 학습/평가/검증 기간을 거치게 되고 엑셀 sheet에도 주별로 예측치와 실제치를 저장함
    #2018년 기간 동안 주별 예측을 시행할 때 총 걸린 computation time도 함께 나옴
    start = time.time()

    for a, l, i in zip(range(53),range(53),range(53)):
        week_tr = a+0
        interval_tr = 168
        z = train_index[0]
        b = train_index[1]
        week_vali = l+0
        interval_vali = 168
        j = valid_index[0]
        k = valid_index[1]
        week_test = i+0
        interval = 168
        n = test_index[0]
        m = test_index[1]
        training_week = pm3.iloc[z:b+interval_tr*week_tr,:]
        training_XX = training_week
        training_YY = training_week
        training_XX = training_XX.drop('elec', axis=1)
        training_YY = training_YY['elec']
        validation_week = pm3.iloc[j+interval_vali*week_vali:k+interval_vali*week_vali,:]
        validation_XX = validation_week
        validation_YY = validation_week
        Validation_XX = validation_XX.drop('elec', axis=1)
        Validation_YY = validation_YY['elec']
        test_week_XX = pm3.iloc[n+interval*week_test:m+intereval*week_test,:]
        test_week_YY = pm3.iloc[n+interval*week_test:m+interval*week_test,:]
        test_XX = test_week_XX
        test_XX = test_XX.drop('elec', axis=1)
        test_YY = test_week_YY
        test_YY = test_YY['elec']
        model = Sequential()
        model.add(Dense(42, activation='elu', input_dim=79))
        model.add(Dense(21, activation='elu', input_dim=42))
        model.add(Dense(1, activation='elu'))
        model.compile(optimizer='Adam',loss='mse', metrics=['accuracy'])
        model.fit(training_XX, training_YY, batch_size = BATCH_SIZE, epochs = 10, validation_data = (Validation_XX, Validation_YY), verbose=0)

        pred = model.predict(test_XX)
        pred = list(itertools.chain(*pred))
        pred = np.asarray(pred)*100000
        result = pd.DataFrame({'elecf': pred, 'eleca' : test_YY*100000})
        writer = pd.ExcelWriter('C:/Users/JYW/Desktop/Github/repository/elec/weeklyload_18_model_A_12_adjusted_sample_period_for_explanation.xlsx', engine = 'openpyxl', mode = 'a')
        result.to_excel(writer, sheet_name = 'weeklyload_18')
        writer.save()
        # print(training_XX, training_YY, Validation_XX, Validation_YY, result)

    # end = time.time()
    # print("elapsed time:")
    # print(end - start)

MLP()
