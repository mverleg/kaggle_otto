
Please add commands that are useful to group members here.

The git branch 'main' is where the functional code lives. Develop features on other branches and merge them into 'main' once they are sort of usable by others. If you think others should use them, please describe briefly how to (for example here).

The classification code depends on utils and settings modules as well as data directory. That means you can choose between:
- Run everything from the main folder, e.g. "python random_forest/gogogadgettree.py"
- Add the main folder to your PYTHONPATH
- (Doing something else with symlinks or something, but don't add them to the repository.)

The general steps are now:

    from utils.loading import get_training_data
    from validation.crossvalidate import SampleCrossValidator
    
    train_data, true_classes, features = get_training_data()
    validator = SampleCrossValidator(train_data, true_classes, test_frac = 0.3, use_data_frac = 0.7, show = True)
    for train, classes, test in validator.yield_cross_validation_sets(rounds = 13):
        # your training code
        prediction = # your classification code
        validator.add_prediction(prediction)
    validator.print_results()

Furthermore it is worth noting:
* If needed, normalize the data using utils/normalize.py.
* If you want, create a submission file using utils/ioutils.py and upload it to Kaggle.
These steps will change as code gets added.


