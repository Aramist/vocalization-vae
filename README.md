Gerbilizer VAE

Installation:

Start by cloning and creating a new environment with the necessary pre-reqs:
```
git clone git@github.com:Aramist/vocalization-vae.git
cd vocalization-vae
conda env create --file environment.yml
```

Train a model:
```
python train.py jsons/local.json
```