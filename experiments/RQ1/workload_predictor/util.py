def mean_absolute_percentage_error(test_data, predictions):
    if len(test_data) != len(predictions):
        raise ValueError("The lengths of test_data and predictions must be the same.")
    
    total_error = 0
    max_error = 0
    
    for i in range(len(test_data)):
        if test_data[i] == 0:
            continue
        error = abs((test_data[i] - predictions[i]) / test_data[i])
        total_error += error
        if error > max_error:
            max_error = error
    
    if len(test_data) > 0:
        mape = total_error / len(test_data) * 100
    else:
        mape = 0
    return mape, max_error * 100