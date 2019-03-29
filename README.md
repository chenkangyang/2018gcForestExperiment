

> Do the experiment on 2017, 2018, 2019 GECCO water detection


## Environment

```
# check out the virtual env in anaconda 

conda info -e

# create virtual env "gc"

conda create -n gc -y python=3.6 jupyter

#install all the requirements in the vitual env

pip install -r requirements.txt

# activate virtual env named "gc"：
source activate gc

# add the kernal in anaconda python3.6 to jupyter notebook

https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook

# activate env "gc"
conda install nb_conda

# then the "gc" kernal will be added to jupyter notebook


# if you do not use virtual env (fix pip install `s permission denied problem):
# pip install --user [package]

```
 - Data download: [pan.baidu](https://pan.baidu.com/s/1i3qPGwqxUXHuXndZ6bKdHA) , password: 8jji
```
.
+-- data
|   +-- water
|   +-- adult
+-- examples
|   +-- logs
|   +-- xxx.py
+-- gcforest
|   +-- __init__.py
|   +-- ...
+-- _notebook
|   +-- 
+-- pkl
|   +-- xx.pkl
+-- README.md
+-- requirements.txt
```