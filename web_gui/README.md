# Instructions to run the app locally

## On Windows

Ensure Python 3.8.6 is installed

Then run the following in the web_gui directory. Note that if pip3 is used, replace `pip` with `pip3`

1. Setup the python virtual environment and install the dependencies. This is done in the directory with the `requirements.txt` file.
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   pip install install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
   ```
2. Change directory to the `src` directory and start the app
   ```bash
   cd src
   flask run
   ```

## On Linux

1. Install python 3.8
   ```bash
   sudo dnf install python3.8 -y
   ```
2. Setup the python virtual environment and install the dependencies. This is done in the directory with the `requirements.txt` file.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip3 install -r requirements.txt
   pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
   ```
3. Change directory to the `src` directory and start the app
   ```bash
   cd src
   flask run
   ```
