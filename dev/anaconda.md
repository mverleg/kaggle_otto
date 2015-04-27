
Installation Anaconda
-------------------------------

Note that these steps are for Anaconda. If you do not use Anaconda, see dev/pip.md .

After installing Anaconda, make sure your scikit is the latest version by typing: 

    conda install -f scikit-learn 
    
Instructions for Lasagne are only available for pip so far, so you can't run neural network code.

If you want to get multiple libraries up to date, this command will install most, if not all, of the requirements for this project.

    easy_install Theano argparse matplotlib mock nose numpy pyparsing python-dateutil pytz scikit-learn scipy six wsgiref