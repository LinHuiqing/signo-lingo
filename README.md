# signo-lingo

# 1. Table of contents <a name="TOC"></a>

1. [Table of contents](#TOC)
2. [Directory structure](#DS)
3. [Final files and instructions on running them](#INSTRUCTIONS)

# 2. Directory structure <a name="DS"></a>

# 3. Final files and instructions on running them <a name="INSTRUCTIONS"></a>

First set up the Conda environment

1. configure the conda enviornment
   ```bash
   conda env create -f environment.yml
   ```
2. To use environment
   ```bash
   conda activate dl-big-proj
   ```
3. Open jupyter noteboook
   ```bash
   jupyter-notebook
   ```
4. To exit environment
   ```bash
   conda deactivate
   ```
5. To destroy envionment
   ```bash
   conda info --envs
   conda env remove --name dl-big-proj
   ```
