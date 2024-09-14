from openai import OpenAI

client = OpenAI(
    api_key="-",
    base_url="https://fridaplatform.com/v1"
)
response = client.chat.completions.create(
  model="tgi",
    messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ],stream=False
)

print(response.choices[0].message.content)