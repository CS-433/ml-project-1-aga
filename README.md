# ML Project 1 - Team AGA

#### Authors: Arda Civelekoglu (311743), Arnault Stähli (313240), Guillen Steulet (316143)

This is the repository for our project 1 for the ML course in EPFL (CS-433).

The relevant data (pushed through Git LFS) can be found in the ```dataset``` folder. Included are the ```x_train.csv```, ```x_test.csv```, and ```y_train.csv``` files.

The provided helpers are found within the ```helpers.py``` file, while the required implementations are found in the ```implementations.py``` file. Our own helpers that we use are found in the ```helpers_create_data.py``` file.

The process of hyperparameter tuning is conducted in the ```Hyperparameter_tuning.ipynb``` notebook, for all feature settings. The ```run_19.ipynb``` and the ```run.ipynb``` notebooks contain the process for the runs for the 19-feature and all-feature settings respectively, the former of which was used for the experimentation with data processing and balancing. Its results are shown in ```Scores.md```. 

Finally, the ```run.ipynb``` file is also the best-performing system that we have developed, which is that of the all-features model (one-hot encoded) with a learning rate gamma of 0.05 and a L2 regularisation lambda of 0. The output from running this code is the best performing result on AIcrowd.
