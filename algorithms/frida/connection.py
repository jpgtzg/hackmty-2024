from openai import OpenAI
import json
client = OpenAI(
    api_key="-",
    base_url="http://198.145.126.109:8080/v1"
)

format = [{"source": "<name of source>","data": [{"level": "<enter level>","units": "<enter unit>"},{"level": "<enter level>","units": "<enter unit>"}]},{"source": "<name of source>","data": [{"level": "<enter level>","units": "<enter unit>"}]}]


response = client.chat.completions.create(
  model="gpt-4o",
    messages=[
    {"role": "system", "content": f"You are a a set of sensors that gather data from a crop. Output a JSON object structured like: {json.dumps(format)}. Make sure to include the source of the data and the data itself. Gather information from at least 4 sources"},
    {"role": "user", "content": "What's the reading for a crop? "},
  ],
  stream=False
)

print(response.choices[0].message.content)