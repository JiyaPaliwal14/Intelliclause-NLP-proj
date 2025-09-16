to start:

in the terminal:
python3 -m venv venv
pip install -r requirements.txt (if later on any error arises then pip install __name_of_the_library__ )

docker compose up -d

uvicorn main:app --reload

keep the terminal running,
in a new terminal

streamlit run frontend.py
