virtualenv env
source env/bin/activate
pip install dvc
pip install "dvc[s3]"
pip install -r ./requirements.txt
wget -O data/raw_data/raw.csv https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv
