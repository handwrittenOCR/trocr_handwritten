import json
import argparse
from os.path import join
import os

from openai import OpenAI
from jinja2 import Template

client = OpenAI()

openai_api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a NER schema with LLM.")
    parser.add_argument("--PATH_DATA", type=str, help="Path to the data files")
    parser.add_argument(
        "--text", type=str, help="name of the text file", default="example_death_act"
    )
    parser.add_argument(
        "--PATH_CONFIG", type=str, help="Path to the config file, schema and PROMPT"
    )
    parser.add_argument(
        "--prompt", type=str, help="name of the prompt file", default="death_act"
    )
    parser.add_argument(
        "--schema", type=str, help="name of the schema file", default="death_act_schema"
    )

    args = parser.parse_args()

    text = open(join(args.PATH_DATA, f"{args.text}.txt"), "r").read()
    text = text.replace("(", "").replace(")", "")

    prompt = Template(open(join(args.PATH_CONFIG, f"{args.prompt}.prompt"), "r").read())
    content = prompt.render(text=text)

    with open(join(args.PATH_CONFIG, f"{args.schema}.json")) as file:
        schema = json.load(file)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ],
        tools=[{"type": "function", "function": schema}],
        tool_choice="auto",
        temperature=0,
        top_p=1.0,
    )

    print(response.choices[0].message.tool_calls[0].function.arguments)
