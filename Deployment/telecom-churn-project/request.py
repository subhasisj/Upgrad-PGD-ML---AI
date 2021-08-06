import requests
data = {
    "total_ic_mou_8": 5.49,
    "total_rech_amt_diff": -206,
    "total_og_mou_8": 0,
    "arpu_8": 656.3919999999999,
    "roam_ic_mou_8": 0,
    "roam_og_mou_8": 0,
    "std_ic_mou_8": 0,
    "av_rech_amt_data_8": 708.893089087563,
    "std_og_mou_8": 0
  }

url = "http://127.0.0.1:5000/predict_api"
response = requests.post(url, json=data)
print("Churn: "+ str(response.json()))
