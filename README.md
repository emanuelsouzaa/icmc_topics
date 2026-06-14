## Comandos para serem executados de acordo com o sistema operacional.

Linux

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 run_all.py

Windows

python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\activate
pip install -r requirements.txt
python run_all.py