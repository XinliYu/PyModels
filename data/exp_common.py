from os import path
from datetime import datetime
from typing import Tuple

from info import log_folder_name
from util.io_ext import pickle_save
from util.general_ext import kv_list_format, error_print, list2d, exec_method
from util.time_exp import tic, toc
import numpy as np
import sys
import os


def log_file_path(data_folder: str, extension: str = '.txt'):
    return path.join(data_folder, log_folder_name, datetime.now().strftime('%m%d%Y_%H%M%S') + extension)


def time_stamp_for_filename():
    return datetime.now().strftime('%m%d%Y_%H%M%S')


def _to_filename_str(val):
    return int(val * 1000) if type(val) in (float, np.float16, np.float32, np.float64) else val


def _merge_name_tuples(default_name_tuple, extra_name_or_names):
    if extra_name_or_names is not None:
        if type(extra_name_or_names) is str:
            return (extra_name_or_names,) + extra_name_or_names
        elif type(extra_name_or_names) is tuple:
            return extra_name_or_names + default_name_tuple
        elif type(extra_name_or_names) is list:
            return tuple(extra_name_or_names) + default_name_tuple
    else:
        return default_name_tuple


def train_model(model, training_set, targets, extra_arguments: Tuple = None, train_method_names=None, error_tag: str = None, error_msg: str = None):
    """
    Simple method for convenience to call the training method of a model where the name of the training method is not exactly known.
    :param model: the model to train.
    :param training_set: provides the training set.
    :param targets: provides the training targets.
    :param extra_arguments: provides extra training arguments.
    :param train_method_names: provides potential training method names. Note 'fit' and 'train' will always be checked by this method, so no need to provide them in this argument.
    :param error_tag: provides a tag for the error message if none of the `train_method_names` or 'fit' or 'train' is found as a callable method of `model`.
    :param error_msg: provides an error message the will be printed to the console following the `error_tag`.
    :return: what the training method returns.
    """
    args = (training_set, targets)
    if extra_arguments is not None:
        args += extra_arguments
    methods = _merge_name_tuples(('fit', 'train'), train_method_names)
    return exec_method(obj=model,
                       callable_names=methods,
                       arguments=args,
                       error_tag=train_model.__name__ if error_tag is None else error_tag,
                       error_msg=error_msg)


def test_model(model, test_set, extra_arguments: Tuple = None, metric=None, targets=None, prediction_method_names=None, error_tag: str = None, error_msg: str = None):
    """
    Simple method for convenience to call the prediction method of a model where the name of the training method is not exactly known.
    In addition, we evaluate a metric for the test result if `targets` and `metric` are provided.
    :param model: the model to test.
    :param test_set: provides the test set.
    :param extra_arguments: provides extra training arguments.
    :param metric: provides the metric to evaluate prediction results.
    :param targets: provides the test targets to evaluate the prediction results.
    :param prediction_method_names: provides potential prediction method names. Note 'predict' and 'predict_proba' will always be checked by this method, so no need to provide them in this argument.
    :param error_tag: provides a tag for the error message if none of the `prediction_method_names` or 'fit' or 'train' is found as a callable method of `model`.
    :param error_msg: provides an error message the will be printed to the console following the `error_tag`.
    :return: what the training method returns.
    """
    args = (test_set,)
    if extra_arguments is not None:
        args += extra_arguments
    methods = _merge_name_tuples(('predict_proba', 'predict'), prediction_method_names)
    predictions = exec_method(obj=model,
                              callable_names=methods,
                              arguments=args,
                              error_tag=test_model.__name__ if error_tag is None else error_tag,
                              error_msg=error_msg)
    return predictions, metric(targets, predictions) if metric is not None and targets is not None else None


def experiment(exp_name: str, para_tune_iter, para_names, model_run, metric_names, metrics, post_model_run, model_run_repeat: int = 3, break_after_first_para_set: bool = False,
               exp_saves_folder: str = None, get_exp_saves=None, exp_save_names=None, save_fun=None, exp_save_file_ext='dat'):
    tic("Experiment {} begins".format(exp_name), key=exp_name)

    metric_count = len(metrics)
    metric_results = list2d(metric_count, model_run_repeat, 0)
    best_metrics = [float('-inf')] * metric_count
    best_para_sets = [None] * metric_count
    _best_metric_keys = ['best ' + metric_name for metric_name in metric_names]
    if exp_saves_folder is None:
        exp_saves_folder = '.'

    for para_setup in para_tune_iter:
        for repeat_idx in range(model_run_repeat):
            model, batch_data, test_loss, test_metric, test_predictions = model_run(para_setup)
            for i in range(metric_count):
                metric_results[i][repeat_idx] = metrics[i](model=model, batch_data=batch_data, test_loss=test_loss, test_metric=test_metric, test_predictions=test_predictions)

            if get_exp_saves is not None:
                things_to_save = get_exp_saves(model, batch_data)
                for save_idx, objs_to_save in enumerate(things_to_save):
                    obj_count = len(objs_to_save)
                    has_save_idx_name = len(exp_save_names) - obj_count
                    timestamp_str = time_stamp_for_filename()
                    for obj_idx in range(obj_count):
                        exp_save_name = exp_save_names[obj_idx + has_save_idx_name]
                        save_file_path = path.join(exp_saves_folder, '{}_{}{}_{}_{}_{}.{}'.format(exp_name, exp_save_names[0] if has_save_idx_name else '', save_idx,
                                                                                                  kv_list_format(keys=metric_names, values=metric_results, value_idx=repeat_idx,
                                                                                                                 value_transform=_to_filename_str,
                                                                                                                 kv_delimiter='', pair_delimiter='_'),
                                                                                                  exp_save_name, timestamp_str, exp_save_file_ext))
                        if save_fun is None:
                            pickle_save(file_path=save_file_path, data=objs_to_save[obj_idx], compressed=False)
                        else:
                            try:
                                if objs_to_save[obj_idx] is not None:
                                    save_fun(save_file_path, objs_to_save[obj_idx])
                                    print("file saved for {} at {}.".format(exp_save_name, save_file_path))
                                else:
                                    print("file not saved for {} because it is None.".format(exp_save_name))
                            except Exception as e:
                                if path.exists(save_file_path):
                                    os.remove(save_file_path)
                                print("file not saved for {} because '{}'.".format(exp_save_name, e))

            post_model_run()

        avg_metrics = [0] * metric_count
        para_setup_str = kv_list_format(keys=para_names, values=para_setup)
        for i in range(metric_count):
            avg_metrics[i] = np.mean(metric_results[i]).round(5)
            if best_metrics[i] < avg_metrics[i]:
                best_metrics[i] = avg_metrics[i]
                best_para_sets[i] = para_setup_str

        print(', '.join([para_setup_str, kv_list_format(keys=metric_names, values=avg_metrics), kv_list_format(keys=_best_metric_keys, values=best_metrics)]))
        sys.stdout.flush()
        if break_after_first_para_set:
            break

    print('\n')
    print('-------------best parameter setup for each metric-------------')
    for i, metric_name in enumerate(metric_names):
        print('{}. for {}'.format(i, metric_name))
        print(best_para_sets[i])

    toc("Experiment {} done".format(exp_name), key=exp_name)
