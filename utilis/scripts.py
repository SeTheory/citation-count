import json
import math

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def eval_result(all_true_values, all_predicted_values):
    # print(mean_absolute_error(all_true_values, all_predicted_values, multioutput='raw_values'))
    mae = mean_absolute_error(all_true_values, all_predicted_values)
    mse = mean_squared_error(all_true_values, all_predicted_values)
    rmse = math.sqrt(mse)
    r2 = r2_score(all_true_values, all_predicted_values)

    # print('true', all_true_values[:10])
    # print('pred', all_predicted_values[:10])
    return [mae, r2, mse, rmse]


def result_format(results):
    loss, mae, r2, mse, rmse = results
    format_str = '| loss {:8.3f} |\n' \
                 '| MAE {:9.3f} | R2 {:10.3f} |\n' \
                 '| MSE {:9.3f} | RMSE {:8.3f} |\n'.format(loss, mae, r2, mse, rmse) + \
                 '-' * 59
    print(format_str)
    return format_str


def get_configs(data_source, model_list):
    fr = open('./configs/{}.json'.format(data_source))
    configs = json.load(fr)
    full_configs = {'default': configs['default']}
    for model in model_list:
        full_configs[model] = configs['default'].copy()
        if model in configs.keys():
            for key in configs[model].keys():
                full_configs[model][key] = configs[model][key]
    return full_configs
