virtualenv env
source env/bin/activate
pip install dvc
pip install "dvc[s3]"
pip install -r ./requirements.txt