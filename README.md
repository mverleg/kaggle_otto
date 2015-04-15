
Please add commands that are useful to group members here.

The git branch 'main' is where the functional code lives. Develop features on other branches and merge them into 'main' once they are sort of usable by others. If you think others should use them, please describe briefly how to (for example here).

The general steps (of which you can skip any unnecessary ones):
1. Load train or test data using functions in utils/loading.py (or utils/ioutils.py).
2. If needed, normalize the data using utils/normalize.py.
3. If needed, shuffle the data using utils/shuffling.py (can be unshuffled at the end).
4. Then run your analysis, generating a prediction matrix. (You can normalize with validation/normalize.py).
5. Check the loss, and possibly accuracy, using validation/score.py
6. If you want, create a submission file using utils/ioutils.py and upload it to Kaggle.
These steps will change as code gets added.


