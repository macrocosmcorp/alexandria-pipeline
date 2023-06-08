# chmod +x ./setup.sh
# git clone -b caselaw https://github.com/macrocosmcorp/alexandria-pipeline.git
# wget -O datasets/ill_caselaw.parquet https://www.dropbox.com/s/l8c4jf4wfn0jpbm/ill_caselaw.parquet?dl=1
# python run.py --type text --dataset_path datasets/ill_caselaw.parquet

# sudo apt install git-lfs
# git lfs install
# git lfs pull
python -m pip install -r requirements.txt
python run_extra.py --type text --dataset_path datasets/ill_caselaw.parquet

# nvidia-smi -l 1