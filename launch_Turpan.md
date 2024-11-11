# Launching the Trocr Handwritten project on Turpan
## A. First time setup:
In $HOME directory:

1. Import Python 3.11:


'''bash
   cd $HOME
   module purge
   module load conda/22.11.1
   conda create -n python-3.11 python=3.11
   conda activate python-3.11

'''


2.  Download and install Poetry:

'''bash
   curl -sSL https://install.python-poetry.org | python3 -
'''


Add Poetry to your PATH (add this to your .bashrc or .bash_profile):

'''bash
   export PATH="$HOME/.local/bin:$PATH"
'''

In $WORK/trocr_handwritten directory:

3. Create a virtual environment using Poetry:


'''bash
   cd $WORK/trocr_handwritten
   poetry config virtualenvs.in-project true
   poetry env use $(which python)
   poetry install

   '''

## B. To reconnect and activate the previously created environment:

1. Activate Conda and Python 3.11 (for Turpan only):

'''bash
   cd $HOME
   module purge
   module load conda/22.11.1
   conda activate python-3.11
'''

2. Navigate to your project and activate the Poetry environment (en local):

'''bash
   cd $WORK/trocr_handwritten
   poetry shell
'''
