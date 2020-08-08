import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'displacement':307, 'horsepower':130, 'weight':3504, 'origin_European':0, 'origin_Japanese':0, 'model year_76':0, 'model year_77':0, 'model year_78':0, 'model year_79':0, 'model year_80':0, 'model year_81':0, 'model year_82':0})

print(r.json())