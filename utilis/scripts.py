import math

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def eval_result(all_true_values, all_predicted_values):
    # print(mean_absolute_error(all_true_values, all_predicted_values, multioutput='raw_values'))
    mae = mean_absolute_error(all_true_values, all_predicted_values)
    mse = mean_squared_error(all_true_values, all_predicted_values)
    rmse = math.sqrt(mse)
    r2 = r2_score(all_true_values, all_predicted_values)
    return [mae, r2, mse, rmse]


def result_format(results):
    loss, mae, r2, mse, rmse = results
    format_str = '| loss {:8.3f} |\n' \
                 '| MAE {:9.3f} | R2 {:10.3f} |\n' \
                 '| MSE {:9.3f} | RMSE {:8.3f} |\n'.format(loss, mae, r2, mse, rmse) + \
                 '-' * 59
    print(format_str)
    return format_str
