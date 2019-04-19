#!/usr/bin/env python3
"""
Hyperparameter tuning using Nevergrad (Facebook gradient-free optimization library)

Since Nevergrad performs minimization, we will chose to minimize:
    f(x) = 1 - max accuracy on validation data with hyperparameters x

To install (see https://github.com/facebookresearch/nevergrad):
    pip install --user nevergrad
"""
import sys
import time
import signal
import random
import subprocess
import numpy as np
import nevergrad.optimization as optimization

from datetime import datetime
from nevergrad import instrumentation as inst

from file_utils import get_num_finished, get_average_valid_accuracy, get_last_int
from pickle_data import load_pickle, save_pickle
from hyperparameter_tuning_commands import output_command


class Network:
    """
    Run the neural net and return (1 - highest validation accuracy)

    Note: this assumes it's running on the same filesystem as training (and in
    the same directory) since it watches for files in the training log directory.
    """
    def __init__(self, instrum, params):
        args, _ = instrum.data_to_arguments(params, deterministic=True)
        self.instrum = instrum
        self.params = params
        self.args = args
        self.jobs = []  # slurm job IDs

        # What command we'll run
        name, train_command, _ = output_command(*args)
        self.name = name
        self.train_command = train_command

        # Where the log files will be stored (synced with rsync, see Sync())
        # Warning: must be set to the same as in kamiak_config.sh
        self.log_dir = "kamiak-logs-" + name

        # The start script creates 3 separate jobs, so once we have 3 "finished"
        # files we're done.
        self.num_folds = 3

    def start(self):
        """ Start the training by submitting to the queue """
        # Get the response to know what job number these are
        cmd = self.train_command.split(" ")
        output = subprocess.check_output(cmd)
        output = output.decode("utf-8").strip().split("\n")

        for line in output:
            self.jobs.append(get_last_int(line))

    def finished(self):
        """ Check if the file saying we're done exists """
        return get_num_finished(self.log_dir) == self.num_folds

    def result(self):
        """ Check the result in the best accuracy file """
        best_accuracy = get_average_valid_accuracy(self.log_dir)

        if best_accuracy is not None:
            # Nevergrad performs minimization, but we want to maximize
            # the accuracy
            return 1.0 - best_accuracy

        return None

    def stop(self):
        """ Cancel the slurm jobs """
        if len(self.jobs) > 0:
            cmd = ["scancel"]+[str(j) for j in self.jobs]
            output = subprocess.check_output(cmd).decode("utf-8")
            print(output, file=sys.stderr)


class GracefulKiller:
    """
    Know when we want to exit gracefully
    https://stackoverflow.com/a/31464349/2698494
    """
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("Exiting gracefully", file=sys.stderr)
        self.kill_now = True


def num_pending(status="pending", username="garrett.wilson"):
    """ Get the number of pending slurm jobs on a user's account """
    cmd = [
        "squeue",
        "-u", username,
        "-t", status,
        "-h"
    ]

    output = subprocess.check_output(cmd)
    output = output.decode("utf-8").strip()

    if output == "":
        return 0

    return len(output.split("\n"))


def make_repeatable():
    """ Set random seeds for making this more repeatable """
    random.seed(1234)
    np.random.seed(1234)


def get_summary(instrum, params=None):
    """ Get human-readable arguments to run with given parameters"""
    # If None, then do default params
    if params is None:
        dim = instrum.dimension
        params = [0] * dim

    return instrum.get_summary(params)


def make_instrumentation(debug=False):
    """ Create the possibilities for all of our hyperparameters """
    # Possible argument values
    batch = inst.var.OrderedDiscrete([2**i for i in range(7, 13)])  # 128 to 4096 by powers of 2
    lr = inst.var.OrderedDiscrete([10.0**(-i) for i in range(3, 6)])  # 0.001 to 0.00001 by powers of 10
    balance = inst.var.OrderedDiscrete([True, False])  # boolean
    units = inst.var.OrderedDiscrete([i*10 for i in range(1, 21)])  # 10 to 200 by 10's
    layers = inst.var.OrderedDiscrete(list(range(1, 13)))  # 1 to 12
    dropout = inst.var.OrderedDiscrete([5*i/100 for i in range(0, 11)])  # 0.0 to 0.5 by 0.05's

    # Our "function" (neural net training with output of max validation accuracy)
    # is a function of the above hyperparameters
    instrum = inst.Instrumentation(batch, lr, balance, units, layers, dropout)

    if debug:
        # Make sure defaults are reasonable and in the middle
        print("Default values")
        print(get_summary(instrum))

    return instrum


def tell_optim(optim, n, jobs_left):
    """ Tell the optimizer the result of a trained network """
    result = n.result()

    if result is not None:
        optim.tell(n.params, result)
        print("Result:", n.name, result,
            "("+str(jobs_left)+" jobs left)", file=sys.stderr)
    else:
        print("Warning:", n.name, "result is None",
            file=sys.stderr)


def hyperparameter_tuning(tool="PortfolioDiscreteOnePlusOne", budget=600,
        num_workers=6, pickle_file="nevergrad_optim.pickle"):
    """
    Run hyperparameter tuning
    See: https://github.com/facebookresearch/nevergrad/blob/master/docs/machinelearning.md

    Note: num_workers is the number of sets of hyperparameters being tuned at a
    time, but the total number of jobs running at a time will be num_workers*3
    since it does 3-fold cross validation.

    It is dynamically updated, so probably set it a bit lower than you want it.
    It'll keep increasing it till there is at least one job always pending.
    But, set it to approximately the number you expect since it might be used
    in the optimization algorithm.
    """
    instrum = make_instrumentation()

    # Optimization -- load if it exists, otherwise create it
    # See: https://github.com/facebookresearch/nevergrad/issues/49
    data = load_pickle(pickle_file)

    if data is not None:
        optim, _ = data

        # Update options if it changed
        optim.budget = budget
        optim.num_workers = num_workers
    else:
        optim = optimization.registry[tool](dimension=instrum.dimension,
            budget=budget, num_workers=num_workers)

    # How many we've done (will stop after "budget" number of them) and the
    # currently training networks
    jobs = 0
    running = []

    # Exit gracefully, cancelling all the training jobs
    killer = GracefulKiller()

    # We can run at most for 7 days
    started = datetime.now()
    time_expired = False

    # Run optimization
    while not killer.kill_now and (jobs < budget or len(running) > 0):
        # If any are done, tell the result and remove it
        #
        # Have to copy since we're removing items from what we're iterating over
        # https://stackoverflow.com/a/5401723/2698494
        running_orig = list(running)

        for n in running_orig:
            if n.finished():
                tell_optim(optim, n, budget-jobs)
                running.remove(n)

        # If we don't have enough running jobs, start more
        while not time_expired and jobs < budget and len(running) < num_workers:
            # New network to train
            n = Network(instrum, optim.ask())

            # Skip if already done (tell though), otherwise start it
            if n.finished():
                print("Already done:", n.name, file=sys.stderr)
                tell_optim(optim, n, budget-jobs)
            else:
                print("Starting:", n.name, file=sys.stderr)
                n.start()
                running.append(n)

            # We've just added another job
            jobs += 1

        # Wait a bit before checking again (the jobs take on the order
        # of hours), but only if we're not supposed to exit
        if not killer.kill_now:
            time.sleep(5)

        # Determine if we should stop queueing jobs since we're running out of
        # time to continue training. Max runtime is 7 days, so stop queueing jobs
        # 5 hours prior (i.e. 6 days 24 hours - 5 hours = 6 days 19 hours)
        time_diff = datetime.now() - started

        if time_diff.days > 6 and time_diff.seconds > 68400:
            print("Time expired", file=sys.stderr)
            time_expired = True

        # Quit if none running and the time has expired
        if time_expired and len(running) == 0:
            break

        # Dynamically change how many jobs we run -- we'd like only a few to be
        # pending at a time
        pending = num_pending()

        # Increase till we have some pending
        if pending == 0:
            num_workers += 1
        # Decrease if we have many pending but with the caveat that it takes
        # a bit of time to get them started, so make sure we never go below 1
        elif pending > 3 and num_workers > 1:
            num_workers -= 1

        # Try to keep relatively real-time for debugging
        sys.stdout.flush()
        sys.stderr.flush()

    # Stop all jobs if killed
    if killer.kill_now:
        for n in running:
            n.stop()

    # Save present optimizer state and current recommendation
    recommendation = optim.provide_recommendation()
    save_pickle(pickle_file, (optim, recommendation), overwrite=True)

    # Get best recommended parameters
    print("Recommendation")
    print(get_summary(instrum, recommendation))
    args, _ = instrum.data_to_arguments(recommendation, deterministic=True)
    _, train, _ = output_command(*args)
    print("Command:", train)

    # We want to exit 1 so we know it's failed to complete
    if killer.kill_now:
        sys.exit(1)


if __name__ == "__main__":
    make_repeatable()
    hyperparameter_tuning()
