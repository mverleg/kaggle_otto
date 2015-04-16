
Please add commands that are useful to group members here.

The git branch 'main' is where the functional code lives. Develop features on other branches and merge them into 'main' once they are sort of usable by others. If you think others should use them, please describe briefly how to (for example here).

The classification code depends on utils and settings modules as well as data directory. That means you can choose between:
- Run everything from the main folder, e.g. "python random_forest/gogogadgettree.py"
- Add the main folder to your PYTHONPATH
- (Doing something else with symlinks or something, but don't add them to the repository.)

The general steps (of which you can skip any unnecessary ones):
1. Load train or test data using functions in utils/loading.py (or utils/ioutils.py).
2. If needed, normalize the data using utils/normalize.py.
3. If needed, shuffle the data using utils/shuffling.py (can be unshuffled at the end).
4. Run your analysis, generating a prediction matrix. (You can normalize with validation/normalize.py).
5. Check the loss, and possibly accuracy, using validation/score.py.
6. If you want, create a submission file using utils/ioutils.py and upload it to Kaggle.
These steps will change as code gets added.


