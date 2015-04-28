
Installation PIP
-------------------------------

Note that these steps do not apply if you are using Anaconda, see `dev/anaconda.md` for that.

All these commands are to be typed in a terminal. In Windows, open `git bash` (by typing `git bash` after pressing Windows key).

If you're at the university (or some other computer where you're now admin), or you want to keep this project seperated from others, you can first do this:

    virtualenv env
    . env/bin/activate

If you use this way, everything is installed locally in directory "env". You need to repeat that last command (activate) every time. If you use an IDE you may need to set it for that one (in PyCharm it's under settings > interpreter).

The following steps apply with or without virtualenv. To install the Python packages that you need, you can do:

    pip install -r dev/freeze.pip

After that, if you want to do any neural network stuff, you need Lasagne. Just type:

    pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt

Check carefully for errors in the middle. If there are none, you are done!

