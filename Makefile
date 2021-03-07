run-api:
	FLASK_APP=prediction_api/api.py	flask run

run-redis:
	sudo docker run --network host redis
