from openai import OpenAI
import json
from flask import Flask, request, jsonify, request

app = Flask(__name__)

client = OpenAI(
    api_key="-",
    base_url="http://198.145.126.109:8080/v1"
)

#format = [{"source": "<name of source>","data": [{"level": "<enter level>","units": "<enter unit>"},{"level": "<enter level>","units": "<enter unit>"}]},{"source": "<name of source>","data": [{"level": "<enter level>","units": "<enter unit>"}]}]

@app.route('/crops', methods=['GET'])
def get_crops_recommendation():
    system_message = "You are an agricultural assistant that, based off of sensor data provided, will give the user a recommendation based off the sensor data provided"
    data = request.get_json()  
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": json.dumps(data)},
        ],
        max_tokens=1024,  # Increase token limit to allow longer responses
        stream=False
    )
    return response.choices[0].message.content

@app.route('/crops', methods=['GET'])
def get_management_recommendation():
    system_message = "You are a storage assistant that, based off of visual data received, will give the company a recomendation on how to manage the storage of the crops. If the crops are in a good state, you will recommend to keep them in storage. If the crops are in the process of spoiling, you will recommend to sell them. If the crops are already spoiled, you will recommend to dispose of them."
    data = request.get_json()  
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": json.dumps(data)},
        ],
        max_tokens=1024,  # Increase token limit to allow longer responses
        stream=False
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    app.run(port=8080)