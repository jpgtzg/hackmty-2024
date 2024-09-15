from openai import OpenAI
import json
from flask import Flask, request, jsonify, request

app = Flask(__name__)

client = OpenAI(
    api_key="-",
    base_url="http://198.145.126.109:8080/v1"
)

@app.route('/getCrops_text', methods=['GET'])
def get_crops_recommendation():
    recommended_time = get_crops_time_recommendation(request.get_json())

    system_message = f"You are an agricultural assistant that, based off of sensor data provided, will give the user a recommendation based off the sensor data provided, and the new time for the crops to be watered: {recommended_time}."
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

    sprinklers = next(item['sprinklers'] for item in data if 'sprinklers' in item)

    recommended_time = [recommended_time] * sprinklers

    return jsonify({
        "recommended_time": recommended_time,
        "response": response.choices[0].message.content
    })

@app.route('/getCrops_time', methods=['GET'])
def get_crops_time_recommendation(function_data=None):
    system_message = "You are an agricultural assistant that, based off of sensor data provided, will give the user ONLY a new time OR COMMA SEPARATED TIMES IF USEFUL for the crops to be watered, NO TEXT"
    
    data = None
    if function_data is None:
        data = request.get_json()
    else:
        data = function_data
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": json.dumps(data)},
            {"role": "assistante", "content": "HH:MM"},
        ],
        max_tokens=1024,  # Increase token limit to allow longer responses
        stream=False
    )
    return response.choices[0].message.content

@app.route('/management', methods=['GET'])
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