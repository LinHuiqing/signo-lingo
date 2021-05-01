# Instructions to run the app locally

## On Windows

Ensure Python 3.8.6 is installed

Then run the following in the web_gui directory. Note that if pip3 is used, replace `pip` with `pip3`

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
