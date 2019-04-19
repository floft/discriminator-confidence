"""
Generate a bunch of hyperparameter values that we'll then queue to run
"""
import math


def output_command(batch, lr, balance, units, layers, dropout, test=False):
    if balance:
        balance = "--balance"
        balance_short = "b"
    else:
        balance = "--nobalance"
        balance_short = "nb"

    name = "cv-" \
        + "b"+str(int(math.log2(batch))) + "-" \
        + "l"+str(int(-math.log10(lr))) + "-" \
        + balance_short + "-" \
        + "u"+str(units) + "-" \
        + "l"+str(layers) + "-" \
        + "d"+str(int(dropout*100))

    args = "--dataset=al.zip " \
        + "--features=al " \
        + "--units="+str(units) + " " \
        + "--layers="+str(layers)

    # Train on train+valid and evaluate on test
    test_str = " --test" if test else ""
    # Since we evaluate on test, don't pick the best on it (i.e. cheating) but
    # just take the last model
    test_eval_str = " --last" if test else ""

    train_args = args + " " \
        + "--model=flat " \
        + "--batch="+str(batch) + " " \
        + "--lr=%.5f "%lr \
        + balance + " " \
        + "--dropout=%.2f"%dropout + test_str
    eval_args = args + test_eval_str

    train = "./kamiak_queue_all_folds.sh " + name + " " + train_args
    evaluate = "sbatch kamiak_eval.srun " + name + " " + eval_args

    return name, train, evaluate
