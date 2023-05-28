# chmod +x ./setup.sh
# git clone https://github.com/macrocosmcorp/alex.git

sudo apt install git-lfs
git lfs install
git lfs pull
python -m pip install -r requirements.txt
python run_extra.py --type title --test

# nvidia-smi -l 1