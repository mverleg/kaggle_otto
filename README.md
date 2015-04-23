
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

To install the Python packages that you need, you can do:

    install -r dev/freeze.pip

After that you need Lasagne. In an unrelated directory (not the project and not site-packages), type:

    git clone https://github.com/Lasagne/Lasagne.git
    cd Lasagne
    python setup.py install

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
    validator = SampleCrossValidator(train_data, true_classes, test_frac = 0.3)
    for train, classes, test in validator.yield_cross_validation_sets(rounds = 13):
        # your training code
        prediction = # your classification code
        validator.add_prediction(prediction)
    validator.print_results()

Furthermore it is worth noting:
* If needed, normalize the data using utils/normalize.py.
* If you want, create a submission file using utils/ioutils.py and upload it to Kaggle.
These steps will change as code gets added.


