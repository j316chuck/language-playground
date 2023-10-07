import boto3
import json

bedrock = boto3.client(service_name="bedrock-runtime")
body = json.dumps(
    {
        "prompt": "\n\nHuman: Tell me a funny joke about outer space\n\nAssistant:",
        "max_tokens_to_sample": 100,
    }
)

response = bedrock.invoke_model(body=body, modelId="anthropic.claude-v2")

response_body = json.loads(response.get("body").read())
print(response_body.get("completion"))