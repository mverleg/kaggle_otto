
The git branch 'main' is where the functional code lives. Develop features on other branches and merge them into 'main' once they are sort of usable by others. If you think others should use them, please describe briefly how to (for example here).

Git instruction
-------------------------------

* git clone [path] - download a git directory you don't have yet
* git pull -u origin [branch] - get the changes from other people for branch
* git branch - see all the branches (current one has a *)
* git checkout [branch] - go to another branch (commit your changes first)
* git status - shows which files have been added and committed
* git commit -m "[message]" - save all the added file as commit (which is like a chapter)
* git push -u origin [branch] - send your commits (the opposite of push)
* git merge [branch] - add everything from the named branch to the current branch

Installation
-------------------------------

If you're at the university (or some other computer where you're now admin), or you want to keep this project seperated from others, you can first do this:

    virtualenv env
    . env/bin/activate

If you use this way, everything is installed locally in directory "env". You need to repeat that last command (activate) every time. If you use an IDE you may need to set it for that one (in PyCharm it's under settings > interpreter).

The following steps apply with or without virtualenv. To install the Python packages that you need, you can do:

    pip install -r dev/freeze.pip

After that, if you want to do any neural network stuff, you need Lasagne. Just type:

    pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt

Check carefully for errors in the middle. If there are none, you are done!

How to run
-------------------------------

The classification code depends on utils and settings modules as well as data directory. That means you can choose between:
- Run everything from the main folder, e.g. "python random_forest/gogogadgettree.py"
- Add the main folder to your PYTHONPATH
- (Doing something else with symlinks or something, but don't add them to the repository.)

Implementing classifiers
-------------------------------

The general steps are now:

    from utils.loading import get_training_data
    from validation.crossvalidate import SampleCrossValidator
    
    train_data, true_classes, features = get_training_data()
    validator = SampleCrossValidator(train_data, true_classes, rounds = 5, test_frac = 0.3)
    for train, classes, test in validator.yield_cross_validation_sets():
        # your training code
        prediction = # your classification code
        validator.add_prediction(prediction)
    validator.print_results()

Furthermore it is worth noting:
* If needed, normalize the data using utils/normalize.py.
* If you want, create a submission file using utils/ioutils.py and upload it to Kaggle.
* You should run the script with `-v` parameter to show more output:

    python demo/test_crossvalidate.py -v

Optimizing
-------------------------------

The general steps for parameter optimization are very similar to those for cross validation (including using `-v` for the command):

    from utils.loading import get_training_data
    from validation.crossvalidate import SampleCrossValidator
    from validation.optimize import GridOptimizer
    
    
    train_data, true_classes, features = get_training_data()
    validator = SampleCrossValidator(train_data, true_classes, rounds = 5, test_frac = 0.1, use_data_frac = 0.2)
    optimizer = GridOptimizer(validator = validator, learning_rate = [10, 1, 0.1, 0.01, 0.001], hidden_layer_size = [60, 30, 50, 40, 20], momentum = 0.9, use_caching = True)
    for parameters, train, classes, test in optimizer.yield_batches():
        prediction = # your code here, which should use the specified parameters
        optimizer.register_results(prediction)
    optimizer.print_plot_results()

Note that results are cached in `settings.OPTIMIZE_RESULTS_DIR`. You can turn this off with the `use_caching` parameter, or remove the directory to clear chance.

As suggested by Jona, you can also use hyperopt (http://jaberg.github.io/hyperopt/) to find an optimum automatically. I don't know about performance; the interface is different but understandable and it is more automatic but without the figure.



