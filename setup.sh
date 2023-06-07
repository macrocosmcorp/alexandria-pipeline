# chmod +x ./setup.sh
# git clone -b caselaw https://github.com/macrocosmcorp/alexandria-pipeline.git

# sudo apt install git-lfs
# git lfs install
# git lfs pull
python -m pip install -r requirements.txt
python run_extra.py --type text --dataset_path datasets/ill_caselaw.parquet

# nvidia-smi -l 1