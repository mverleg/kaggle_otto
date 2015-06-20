
Code description
-------------------------------

Altogether we wrote a lot of code at a 5100+ lines and around 400 commits, so it can be daunting to try and understand. Here is a very brief explanation:

* **utils** - code for loading input, saving output, feature generation and other general functionality.
* **validation** - code for cross validation; optimize does a grid search described below, using crossvalidate for scoring
* **demo** - code to demonstrate functionality, like non-automatic tests
* **nnet** - Lasagne/nolearn neural network code; 'grids' and 'optimize' contain the parameter searches, 'scikit' is the extended network object compatible with scikit (non-compatible version in oldstyle) and most other files handle enhancements such as scoring, dynamic parameters, optimizations, combining nets and visualization. There is some code duplication due to job files being kept after running (for future reference of settings). 
* **data** and **results** - in- and output respectively (most not part of repo due to size)

Other directories contain code for other methods used.

Settings are handles in settings.py and other development-related meta files are keps in **dev**. This file contains some instructions on how to use common features, for use by our team.

Installation
-------------------------------

The installation steps for Anaconda are in `dev/anaconda.md` whereas the steps for everything else are in `dev/pip.md`.

How to run
-------------------------------

The classification code depends on utils and settings modules as well as data directory. That means you can choose between:
- Run everything from the main folder, e.g. "python random_forest/gogogadgettree.py"
- Add the main folder to your PYTHONPATH
- (Doing something else with symlinks or something, but don't add them to the repository.)

Comparing parameters
-------------------------------

The steps for comparing parameters have changed, which was needed to make it faster. The order is different but it should be just some copy-paste:

    def train_test(train, classes, test, **parameters):
        # your trainig here
        prediction = get_random_probabilities(sample_count = test.shape[0]) # your prediction here
        return prediction
    
    train_data, true_labels = get_training_data()[:2]
    validator = SampleCrossValidator(train_data, true_labels, rounds = 3, test_frac = 0.2, use_data_frac = 1)
    optimizer = ParallelGridOptimizer(train_test_func = train_test, validator = validator,
        learning_rate = [10, 1, 0.1, 0.01, 0.001],  # these parameters can be replaced by your own
        hidden_layer_size = [60, 30, 50, 40, 20],   # use a list to compare parameters
        weight_decay = 0.1,                         # use a text or number for static parameters
        momentum = 0.9                              # providing static parameters this way is useful for caching
    ).readygo()

The old cross validation and optimization should still work well. The old optimization example is removed because it has no advantages over the new method.

Implementing classifiers
-------------------------------

The general steps are now:

    def train_test(train, classes, test, **parameters):
        prediction = get_random_probabilities(sample_count = test.shape[0])
        return prediction
    
    train_data, true_labels = get_training_data()[:2]
    validator = SampleCrossValidator(train_data, true_labels, rounds = 6, test_frac = 0.1, use_data_frac = 1)
    optimizer = ParallelGridOptimizer(train_test_func = train_test, validator = validator, use_caching = True, process_count = max(cpu_count() - 1, 1),
        learning_rate = [10, 1, 0.1, 0.01, 0.001],
        hidden_layer_size = [60, 30, 50, 40, 20],
        weight_decay = 0.1,
        momentum = 0.9
    ).readygo()

Furthermore it is worth noting:
* If needed, normalize the data using `utils/normalize.py`.
* If you want, create a submission file using `utils/ioutils.py` and upload it to Kaggle.
* You should run any script with `-v` parameter to show more output (of `-vv` for much more):

    python demo/test_crossvalidate.py -v

Extra train data
-------------------------------

To convert the confident part of test data to additional training data, you need to have a prediction file. You can download our best one from Kaggle. The path to this file is given by `TOP_PREDICTIONS` (from `settings.py`), so make sure to place it there. Then use:

    from utils.expand_train import expand_from_test
    bigger_data, bigger_labels = expand_from_test(train_data, true_labels, test_data, confidence = params['test_data_confidence'])

Find a good confidence value (but extra size goes down quickly). Best do this before adding features.

Extra features
-------------------------------

To generate extra features using some defaults, use:

    from utils.features import chain_feature_generators
    train_data, test_data = chain_feature_generators(train_data, true_labels, train_data, extra_features = 57, seed = 0)

It finds the existing features that correlate with the difficult classes, then builds new features from them with several randomly selected operations.

Git instruction
-------------------------------

There are instructions for the Windows git GUI on the gitlab wiki. An overview of commands is in `dev/git.md`.

The git branch 'main' is where the functional code lives. Develop features on other branches and merge them into 'main' once they are sort of usable by others. If you think others should use them, please describe briefly how to (for example here).


