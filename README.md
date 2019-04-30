# Multi-Purposing Domain Adaptation Discriminators for Pseudo Labeling Confidence

Method: instead of using task classifier's softmax confidence for weighting samples for pseudo labeling, use the discriminator's / domain classifier's confidence based on how source-like the feature representations of the samples appear. In other words, we multi-purpose the discriminator to not only aid in producing domain-invariant representations (like in DANN) but also to provide pseudo labeling confidence.

Steps:

- Download and preprocess datasets (*datasets/generate_tfrecords.py*)
- Optionally view the datasets (*datasets/view_datasets.py*)
- Train models (*main.py* or *kamiak_train.srun*)
- Evaluate models (*main_eval.py* or *kamiak_eval.srun*)

## Training
For example, to train on USPS to MNIST with no adaptation:

    ./kamiak_queue.sh test1 --model=vada_small --source=usps --target=mnist --method=none

To pseudo label weighting with the domain classifier's confidence (proposed method) or the task classifier's softmax confidence:

    ./kamiak_queue.sh test1 --model=vada_small --source=usps --target=mnist --method=pseudo
    ./kamiak_queue.sh test1 --model=vada_small --source=usps --target=mnist --method=pseudo --nouse_domain_confidence --debugnum=1

To instead do instance weighting:

    ./kamiak_queue.sh test1 --model=vada_small --source=usps --target=mnist --method=instance
    ./kamiak_queue.sh test1 --model=vada_small --source=usps --target=mnist --method=instance --nouse_domain_confidence --debugnum=1

Or, these but without adversarial training:

    ./kamiak_queue.sh test2 --model=vada_small --source=usps --target=mnist --method=pseudo --nodomain_invariant
    ./kamiak_queue.sh test2 --model=vada_small --source=usps --target=mnist --method=pseudo --nouse_domain_confidence --debugnum=1 --nodomain_invariant

Note: you probably need `--nocompile_metrics` on any SynSigns to GTSRB adaptation, otherwise it may run out of memory. Also, these examples assume you're using SLURM. If not, you can modify *kamiak_queue.sh* to not queue with sbatch but run with bash.

## Evaluating
For example, to evaluate the above "test1" trained models:

    sbatch kamiak_eval.srun test1 --eval_batch=2048 --jobs=1
