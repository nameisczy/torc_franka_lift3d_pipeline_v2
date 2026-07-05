from openai import OpenAI

client = OpenAI(api_key="", base_url="http://lab.cs.lab.edu:4000/v1")

model = client.models.list().model_dump()["data"][0]["id"]

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "this is a test request, write a short poem"}
    ],
)
print(response.choices[0].message.content)
