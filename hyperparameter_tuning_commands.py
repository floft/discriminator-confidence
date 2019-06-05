"""
Generate a bunch of hyperparameter values that we'll then queue to run
"""
import math


def output_command(prefix=None, args=None, lr=None, lr_domain=None, lr_target=None, test=False):
    """
    lr, lr_domain, lr_target -- tuned parameters
    prefix -- which tuning we're doing, e.g. usps-mnist-none
    args -- list of additional arguments, e.g. ["--source=usps",
        "--target=mnist", "--method=none"]
    test -- whether to evaluate only last model or best one (not used for tuning)
    """
    assert prefix is not None
    assert args is not None
    assert lr is not None

    args = " ".join(args)

    if lr_domain is not None:
        lr_domain_mult = lr_domain/lr
    else:
        lr_domain_mult = 1.0
        lr_domain = 1.0  # just for making string 0 in name, not in tune range

    if lr_target is not None:
        lr_target_mult = lr_target/lr
    else:
        lr_target_mult = 1.0
        lr_target = 1.0  # just for making string 0 in name, not in tune range

    name = prefix+"-" \
        + "l"+str(int(-math.log10(lr))) + "-" \
        + "ld"+str(int(-math.log10(lr_domain))) + "-" \
        + "lt"+str(int(-math.log10(lr_target)))
    #+ "b"+str(int(math.log2(batch))) + "-" \

    # Train on train+valid and evaluate on test
    test_str = " --test" if test else ""
    # Since we evaluate on test, don't pick the best on it (i.e. cheating) but
    # just take the last model
    test_eval_str = " --last" if test else ""

    train_args = args + " " \
        + "--model=vada_small " \
        + "--lr=%.5f "%lr \
        + "--lr_domain_mult=%.5f "%lr_domain_mult \
        + "--lr_target_mult=%.5f "%lr_target_mult \
        + test_str
    #+ "--train_batch="+str(batch) + " " \
    eval_args = args + test_eval_str

    train = "./kamiak_queue.sh " + name + " " + train_args
    evaluate = "sbatch kamiak_eval.srun " + name + " " + eval_args

    return name, train, evaluate
