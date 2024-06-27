import random
import itertools
import json
import collections
import sys
import inspect
from tqdm import tqdm
import os
import torch
import logging
import numpy as np
from datasets import Image, Sequence

import lmms_eval.api
import lmms_eval.tasks
from lmms_eval.tasks import get_task_dict,initialize_tasks
import lmms_eval.models
import lmms_eval.api.metrics
import lmms_eval.api.registry
import argparse

from lmms_eval.utils import (
    positional_deprecated,
    run_task_tests,
    make_table,
    create_iterator,
    get_git_commit_hash,
    simple_parse_args_string,
)
os.environ['HF_HOME'] = '/ML-A100/team/mm/zk/cache/datasets'
os.environ['http_proxy'] ='http://100.66.28.72:3128'
os.environ['https_proxy'] ='http://100.66.28.72:3128'

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
        "--output_path",
        default='/ML-A100/team/mm/zk/lmms-eval/logs',
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
args = parser.parse_args()
# @positional_deprecated
def evaluate(
    json_file,
    tasks,
    model,
    output_path,
    filter_difficulty,
    limit=10000,
    bootstrap_iters: int = 100000,
    show_task_to_terminal: bool = False,
    log_samples: bool = False,
    cli_args=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param show_task_to_terminal: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :return
        Dictionary of results
    """
    initialize_tasks()
    args.output_path=output_path

    # json_file='/ML-A100/team/mm/zk/lmms-eval/logs/llava1.5_13b_test_llava/cmmmu_val.json'
    with open(json_file) as json_file:
        responses=json.load(json_file)['logs']


    task_dict = get_task_dict(tasks, model_name=model)
    for task_name in task_dict.keys():  
        task_obj = task_dict[task_name]
        if type(task_obj) == tuple:
            group, task_obj = task_obj
            if task_obj is None:
                continue
    #     # stores the final result for each task, for each metric/filter pair.

    results = collections.defaultdict(dict)
    # Tracks each task's version.
    versions = collections.defaultdict(dict)
    # Tracks the YAML configs of all chosen tasks.
    configs = collections.defaultdict(dict)
    # logs info about each document evaluated.
    samples = collections.defaultdict(list)
    # tracks all Instances/requests a model must generate output on.
    requests = collections.defaultdict(list)
    # Aggregated task scores presented with groups
    results_agg = collections.defaultdict(dict)
    # Aggregated groups scores only
    groups_agg = collections.defaultdict(dict)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = collections.defaultdict(int)
    # store the hierarchy to do proper ordering
    task_hierarchy = collections.defaultdict(list)
    # store the ordering of tasks and groups
    task_order = collections.defaultdict(int)
    task_group_alias = collections.defaultdict(dict)
    # store num-fewshot value per task
    num_fewshot = collections.defaultdict(int)

    # get lists of each type of request
    for task_name, task in task_dict.items():
        if type(task) == tuple:
            group_name, task = task
            task_hierarchy[group_name].append(task_name)
            versions[group_name] = "N/A"

        else:
            group_name = None
            task_hierarchy[task_name] = []

        if task is None:
            continue

        versions[task_name] = task.VERSION
        configs[task_name] = dict(task.dump_config())

        if "num_fewshot" in configs[task_name]:
            n_shot = configs[task_name]["num_fewshot"]
        else:
            n_shot = 0
        num_fewshot[task_name] = n_shot

        if "task_alias" in configs[task_name]:
            task_group_alias[task_name] = configs[task_name]["task_alias"]

        if ("group_alias" in configs[task_name]) and (group_name not in task_group_alias) and (group_name is not None):
            task_group_alias[group_name] = configs[task_name]["group_alias"]

        if limit is not None:
            if task.has_test_docs():
                task_docs = task.test_docs()
            elif task.has_validation_docs():
                task_docs = task.validation_docs()
            else:
                raise RuntimeError("Task has neither test_docs nor validation_docs")
            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

        # task.build_all_requests(limit=limit, rank=lm.rank, world_size=lm.world_size)

    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_name, task in task_dict.items():
        if type(task) == tuple:
            group, task = task
            if task is None:
                continue
        # task.apply_filters()

    ### Collect values of metrics on all datapoints ###
    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for task_name, task in task_dict.items():
        if type(task) == tuple:
            group, task = task
            if task is None:
                continue
        # TODO: make it possible to use a different metric per filter
        samples[task_name]=responses
        docs = task.test_docs() if task.has_test_docs() else task.validation_docs()
        for idx, response in tqdm(enumerate(responses),total=len(responses), desc=f"Postprocessing response"):
            if response['chosen'] in filter_difficulty: #pass useless case
                continue
            doc=response['doc']
            rep=response['filtered_resps']
            metrics = task.process_results(doc, rep)
            for metric, value in metrics.items():
                vals[(task_name, 'none', metric)].append(value)
            ###################### stop print
            print(len(vals[(task_name, 'none', metric)]))

        ### Get task ordering for correct sample-wide aggregation
    group_to_task = {}
    for group in task_hierarchy.keys():
        if group not in task_order:
            task_order[group] = 0

        if len(task_hierarchy[group]) > 0:
            group_to_task[group] = task_hierarchy[group].copy()

        for task in task_hierarchy[group]:
            if task in task_order:
                task_order[task] += 1
            else:
                task_order[task] = 1 + task_order[group]

            if task in task_hierarchy:
                group_to_task[group].remove(task)
                group_to_task[group].extend(task_hierarchy[task])

    task_to_group = {}
    for group in group_to_task:
        for task in group_to_task[group]:
            if task in task_to_group:
                task_to_group[task].append(group)
            else:
                task_to_group[task] = [group]

    ### Aggregate results over all datapoints ###
    # aggregate results ; run bootstrap CIs
    for (task_name, key, metric), items in vals.items():
        # key=str(key)
        task = task_dict[task_name]
        metric_key = metric + "," + key

        if type(task) == tuple:
            group_name, task = task
        else:
            group_name = None

        if metric not in task.aggregation():
            continue

        agg_fn = task.aggregation()[metric]

        # Bo: for models that need to know the args to save to correct path
        if inspect.getfullargspec(agg_fn).args == ["results", "args"]:
            results[task_name][metric_key] = agg_fn(items, args)
        else:
            # Bo: for models only need agg items
            results[task_name][metric_key] = agg_fn(items)

        results[task_name]["samples"] = len(items)

        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this
        if bootstrap_iters > 0:
            stderr = lmms_eval.api.metrics.stderr_for_metric(
                metric=task.aggregation()[metric],
                bootstrap_iters=min(bootstrap_iters, 100) if metric in ["bleu", "chrf", "ter"] else bootstrap_iters,
            )

            if stderr is not None and len(items) > 1:
                results[task_name][metric + "_stderr" + "," + key] = stderr(items)
            else:
                results[task_name][metric + "_stderr" + "," + key] = "N/A"

    if bool(results):
        for group, task_list in reversed(task_hierarchy.items()):
            if task_list == []:
                total_size = results[group]["samples"]
            else:
                total_size = 0

                for task in task_list:
                    metrics = results[task]

                    current_size = metrics.pop("samples")

                    all_stderr = []
                    for metric in [key for key in metrics.keys() if "_stderr" not in key]:
                        stderr = "_stderr,".join(metric.split(","))
                        stderr_score = results[task][stderr]
                        var_score = stderr_score**2 if stderr_score != "N/A" else 0
                        metric_score = results[task][metric]

                        all_stderr.append(stderr)

                        if metric_score is None:
                            results[group][metric] = None
                            results[group][stderr] = 0
                            continue

                        if metric in results[group]:
                            results[group][metric] = (results[group][metric] * total_size + metric_score * current_size) / (total_size + current_size)
                            # $$s_z^2 = \frac{(n-1) s_x^2 + (m-1) s_y^2}{n+m-1} + \frac{nm(\bar x - \bar y)^2}{(n+m)(n+m-1)}.$$
                            results[group][stderr] = ((total_size - 1) * results[group][stderr] + (current_size - 1) * var_score) / (total_size + current_size - 1) + total_size * current_size / (
                                (total_size + current_size) * (total_size + current_size - 1)
                            ) * (results[group][metric] - metric_score) ** 2
                        else:
                            results[group][metric] = metric_score
                            results[group][stderr] = var_score

                    total_size += current_size

                for stderr in all_stderr:
                    results[group][stderr] = np.sqrt(results[group][stderr])

            results[group]["samples"] = total_size

    def print_tasks(task_hierarchy, task_order, task_version, task_group_alias):
        results_agg = collections.defaultdict(dict)
        groups_agg = collections.defaultdict(dict)
        for group_name, task_list in task_hierarchy.items():
            order = task_order[group_name]
            results_agg[group_name] = results[group_name].copy()
            results_agg[group_name]["tab"] = order

            if (order < max(task_order.values())) and (len(task_list) > 0):
                groups_agg[group_name] = results[group_name].copy()
                groups_agg[group_name]["tab"] = order

            if task_list != []:
                for task in sorted(task_list):
                    if task in task_hierarchy:
                        _task_hierarchy = {task: task_hierarchy[task]}
                    else:
                        _task_hierarchy = {task: []}

                    _results_agg, _groups_agg, task_version = print_tasks(_task_hierarchy, task_order, task_version, task_group_alias)

                    results_agg = {**results_agg, **_results_agg}
                    groups_agg = {**groups_agg, **_groups_agg}

        return results_agg, groups_agg, task_version

    results_agg, groups_agg, versions = print_tasks(task_hierarchy, task_order, versions, task_group_alias)

    for task in results_agg:
        task_results = results_agg[task]

        if "samples" in task_results:
            task_results.pop("samples")

        tab_string = ""
        if "tab" in task_results:
            tab = task_results.pop("tab")
            tab_string = " " * tab + "- " if tab > 0 else ""

        if task in task_group_alias:
            task_alias = task_group_alias[task]
            results_agg[task]["alias"] = tab_string + task_alias
        else:
            results_agg[task]["alias"] = tab_string + task

    for group in groups_agg:
        group_results = groups_agg[group]

        if "samples" in group_results:
            group_results.pop("samples")

        tab_string = ""
        if "tab" in group_results:
            tab = group_results.pop("tab")
            tab_string = " " * tab + "- " if tab > 0 else ""

        if group in task_group_alias:
            group_alias = task_group_alias[group]
            groups_agg[group]["alias"] = tab_string + group_alias
        else:
            groups_agg[group]["alias"] = tab_string + group

    for group_name, task_list in task_hierarchy.items():
        if task_list != []:
            num_fewshot[group_name] = num_fewshot[task_list[0]]

    results_dict = {
        'model':model,
        "results": dict(results_agg.items()),
        **({"groups": dict(groups_agg.items())} if bool(groups_agg) else {}),
        "configs": dict(sorted(configs.items())),
        "versions": dict(sorted(versions.items())),
        "n-shot": dict(sorted(num_fewshot.items())),
    }
    if log_samples:
        results_dict["samples"] = dict(samples)
    print(results_dict)
    return results_dict


if __name__=='__main__':
    root_dir='/ML-A100/team/mm/zk/lmms-eval/logs_new/chartqa'
    tasks=['chartqa']
    filter_difficulty=['delete','easy'] #delete easy middle difficult very difficult  总共有多少条
    # filter_difficulty=['delete','easy','middle']
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                model=file.split('.json')[0]
                file_path=os.path.join(root,file)
                evaluate(file_path,tasks,model,root,filter_difficulty)
