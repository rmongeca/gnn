# Log folder for Model runs

This folder will contain the log results and metric outputs for each model run. This subfolders
are ignored in the repository by default.

In the case of iGNNition logs, they are contained within a directory called _Checkpoints_.

In addition, a symbolic link to this direcory is included in each iGNNition example folder, as it
is directly referenced in the _train_options.yml_ file relative to the example folder.
