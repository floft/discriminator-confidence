#!/usr/bin/env python3
"""
Analyze the results from the hyperparameter tuning
"""
import pathlib
import numpy as np
import pandas as pd

from hyperparameter_tuning_commands import output_command


def get_tuning_files(dir_name, prefix="dal_results_cv-"):
    """ Get all the hyperparameter evaluation result files """
    files = []
    matching = pathlib.Path(dir_name).glob(prefix+"*.txt")

    for m in matching:
        name = m.stem.replace(prefix, "")
        file = str(m)
        files.append((name, file))

    return files


def beginning_match(match, line):
    """ Does the first x=len(match) chars of line match the match string """
    return line[:len(match)] == match


def parse_file(filename):
    """
    Get all of the data from the file

    Several parts:
        - Best validation accuracy per target/fold at a particular step
        - target/fold train/test A/B accuracy
        - averages of train/test A/B accuracies
    """
    in_validation = False
    in_traintest = False
    in_averages = False

    validation = []
    traintest = []
    averages = []

    valid_header = "Target,Fold,Model,Best Step,Accuracy at Step"
    traintest_header = "Target,Fold,Train A,Test A,Train B,Test B"
    averages_header = "Dataset,Avg,Std"

    with open(filename) as f:
        for line in f:
            line = line.strip()

            if beginning_match(valid_header, line):
                in_validation = True
                in_traintest = False
                in_averages = False
            elif beginning_match(traintest_header, line):
                in_validation = False
                in_traintest = True
                in_averages = False
            elif beginning_match(averages_header, line):
                in_validation = False
                in_traintest = False
                in_averages = True
            elif len(line) > 0:
                values = line.split(",")

                # For example, if we ran evaluation before we had any models to
                # evaluate, we'd get no data.
                if values[0] == "No data.":
                    return None

                if in_validation:
                    validation.append((values[0], int(values[1]),
                        values[2], int(values[3]), float(values[4])))
                elif in_traintest:
                    traintest.append((values[0], int(values[1]),
                        float(values[2]), float(values[3]), float(values[4]), float(values[5])))
                elif in_averages:
                    averages.append((values[0], float(values[1]), float(values[2])))
            else:
                # Empty lines ends a section
                in_validation = False
                in_traintest = False
                in_averages = False

    validation = pd.DataFrame(data=validation, columns=valid_header.split(","))
    traintest = pd.DataFrame(data=traintest, columns=traintest_header.split(","))
    averages = pd.DataFrame(data=averages, columns=averages_header.split(","))

    return validation, traintest, averages


def compute_mean_std(df, name):
    # ddof=0 is the numpy default, ddof=1 is Pandas' default
    return df[name].mean(), df[name].std(ddof=0)


def compute_val_stats(df):
    return compute_mean_std(df, "Accuracy at Step")


def compute_eval_stats(df):
    names = ["Train A", "Test A", "Train B", "Test B"]
    data = [[name]+list(compute_mean_std(df, name)) for name in names]
    return pd.DataFrame(data=data, columns=["Dataset", "Avg", "Std"])


def parse_name(name):
    # Get values
    values = name.split("-")

    batch = values[0].replace("b", "")
    lr = values[1].replace("l", "")
    balance = values[2]
    units = values[3].replace("u", "")
    layers = values[4].replace("l", "")
    dropout = values[5].replace("d", "")

    # Put into correct ranges
    batch = 2**int(batch)
    lr = 10**(-int(lr))
    balance = balance == "b"
    units = int(units)
    layers = int(layers)
    dropout = int(dropout)/100

    return {
        "batch": batch,
        "lr": lr,
        "balance": balance,
        "units": units,
        "layers": layers,
        "dropout": dropout
    }


def all_stats(files, recompute_averages=False, sort_on_test=False, sort_on_b=False):
    stats = []

    for name, file in files:
        parse_result = parse_file(file)

        if parse_result is None:
            print("Warning: skipping", file)
            continue

        validation, traintest, averages = parse_result

        if recompute_averages:
            averages = compute_eval_stats(traintest)

        validavg = compute_val_stats(validation)

        stats.append({
            "name": name,
            "parameters": parse_name(name),
            "file": file,
            "validation": validation,
            "traintest": traintest,
            "averages": averages,
            "validavg": validavg,
        })

    if sort_on_test:
        # Sort by test accuracy (i.e. cheating)
        stats.sort(key=lambda x: x["averages"][x["averages"]["Dataset"] == "Test A"]["Avg"].values[0])
    elif sort_on_b:
        # Sort by test accuracy on domain B (i.e. cheating)
        stats.sort(key=lambda x: x["averages"][x["averages"]["Dataset"] == "Test B"]["Avg"].values[0])
    else:
        # Sort by validation accuracy
        stats.sort(key=lambda x: x["validavg"][0])

    return stats


def optimize_parameters(stats):
    """
    Optimize each parameter individually, i.e. we're assuming convexity here.

    Pick the best value for each parameter based on what gives the highest
    validation accuracy. Then we'll train that model and see if it does even
    better (it probably won't since the hyperparameter space is probably not
    convex).
    """
    parameter_names = list(stats[0]["parameters"].keys())
    best_params = {}
    best_params_accuracy = {}

    for p in parameter_names:
        param_values = {}

        for s in stats:
            param_value = s["parameters"][p]
            accuracy = s["validavg"][0]

            if param_value not in param_values:
                param_values[param_value] = [accuracy]
            else:
                param_values[param_value].append(accuracy)

        averages = {}

        for param_value, accuracy in param_values.items():
            averages[param_value] = (
                np.array(accuracy).mean(),
                np.array(accuracy).std()
            )

        averages = sorted(list(averages.items()), key=lambda x: x[1][0])
        best = averages[-1]

        best_params[p] = best[0]
        best_params_accuracy[p] = best[1]

    return best_params, best_params_accuracy


if __name__ == "__main__":
    files = get_tuning_files(".")

    # Best on validation data
    best = all_stats(files)[-1]
    print("Best on Validation -", best["name"])
    print(best["averages"])
    print()

    # Best on testing data (i.e. cheating)
    best = all_stats(files, sort_on_test=True)[-1]
    print("Best on Test A (cheating) -", best["name"])
    print(best["averages"])
    print()

    # Best on testing data on domain B (i.e. cheating)
    best = all_stats(files, sort_on_b=True)[-1]
    print("Best on Test B (cheating) -", best["name"])
    print(best["averages"])
    print()

    # Optimize parameters
    stats = all_stats(files)
    best_params, best_params_accuracy = optimize_parameters(stats)
    print("Best Parameters")
    print(best_params)
    print(best_params_accuracy)
    print()

    print("Run with best parameters:")
    command = output_command(
        best_params["batch"],
        best_params["lr"],
        best_params["balance"],
        best_params["units"],
        best_params["layers"],
        best_params["dropout"],
        test=True
    )
    print(command[1])
    print(command[2])
