"""
Note: This script is used to interact with the internal API for
performing inference with various language models.

It will send prompts to the API, retrieve responses, and save them
to specified output files in JSONL format. The script currently supports
inference with 
"""
import polars as pl

import argparse
import json
import os
import requests
import time

def prompt_api(api_key: str,
               prompt: str,
               max_tokens: int,
               temperature: float,
               model: str,
               base_url: str
               ) -> requests.models.Response:
    """
    Send a prompt to the GESIS internal API and return the response.

    Args:
        prompt (str): The prompt to send to the model.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Model temperature.
        model (str): The model to use for inference.

    Returns:
        requests.models.Response: The response from the API.
    """
    # Set headers for the API request.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Set specific data for OpenAI and Ollama models, respectively.
    # Note: Currently, the only difference is the endpoint URL and how
    # max_tokens is specified. We assume more differences may arise
    # in the future, so we keep this separation for now.
    if model in ["o4-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"]:
        data = data_openai(prompt, max_tokens,
                           temperature, model)
    else:
        data = data_ollama(prompt, max_tokens,
                           temperature, model)

    response = requests.post(base_url,
                             headers=headers,
                             json=data)
    
    if not response.ok:
        raise Exception("API request failed with status code "
                        f"{response.status_code}: {response.text}")
    return response

def data_openai(prompt: str,
                max_tokens: int,
                temperature: float,
                model: str,
                ) -> dict:
    """
    Prepare the data for the OpenAI API request.

    Args:
        api_key (str): API key for the API.
        prompt (str): The prompt to send to the model.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Model temperature.
        model (str): The model to use for inference.
        base_url (str): Base URL of the API.

    Returns:
        requests.models.Response: The response from the API.
    """

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,  # we want the full response, not a stream
    }
    return data


def data_ollama(prompt: str,
                max_tokens: int,
                temperature: float,
                model: str,
                ) -> dict:
    """
    Prepare the data for the Ollama API request.

    Args:
        prompt (str): The prompt to send to the model.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Model temperature.
        model (str): The model to use for inference.

    Returns:
        requests.models.Response: The response from the API.
    """
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        # num_predict is used instead of max_tokens in ollama
        "num_predict": max_tokens,  
        "temperature": temperature,
        "stream": False,  # we want the full response, not a stream
    }

    return data

def run_inference(prompt_file: str,
                  output_path: str,
                  api_key: str,
                  base_url: str,
                  repeat: int = 3,
                  max_tokens: int = 100,
                  temperature: float = 0.7,
                  model: str = "llama4:latest",
                  output_file: str = "responses",
                  ) -> None:
    """
    Run inference on prompts from a file and save the responses.

    Args:
        prompt_file (str):
            Path to the file containing prompts.
        output_path (str):
            Directory to save the output files.
        api_key (str):
            API key for the API.
        max_tokens (int):
            Maximum number of tokens to generate.
        temperature (float):
            Temperature for the model.
        model (str):
            Model to use for inference.
        base_url (str):
            Base URL of the API.
        output_file (str):
            Base name for the output files.

    Returns:
        None: The function saves the responses to files.
    """
    start_time = time.time()
    data = pl.read_csv(prompt_file)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for i in range(len(data)):
        print(f"Processing prompt {i + 1}/{len(data)}: {data['id'][i]}")
        prompt = data["prompt"][i]
        prompt_id = data["id"][i]

        for _ in range(repeat):
            response = prompt_api(api_key, prompt, max_tokens,
                                  temperature, model, base_url)
            response_data = response.json()
            
    # Save the raw response in JSONL format
            with open(f"{output_path}/{output_file}.jsonl", "a+") as f:
                response_data["prompt_id"] = prompt_id
                f.write(json.dumps(response_data) + "\n")

    time_taken = time.time() - start_time
    print(f"Responses saved to {output_path}/{output_file}.jsonl")
    print(f"Time taken: {time_taken:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process prompts with "
                                     "the internal LLM API. "
                                     "Note: For now, this is only "
                                     "working for ollama models, not " \
                                     "the OpenAI models provided "
                                     "through the internal API.")
    parser.add_argument("--prompt_file",
                        type=str,
                        required=True,
                        help="Path to the prompt file.")
    parser.add_argument("--base_url",
                        type=str,
                        required=True,
                        help="Base URL of the API.")
    parser.add_argument("--output_path",
                        type=str,
                        default="output",
                        required=True,
                        help="Directory to save the output files.")
    parser.add_argument("--output_file",
                        type=str,
                        default="responses",
                        help="Base name for the output files.")
    parser.add_argument("--api_key",
                        type=str,
                        required=True,
                        help="GESIS internal API key.")
    parser.add_argument("--repeat",
                        type=int,
                        default=3,
                        help="Number of times to repeat the inference "
                        "per prompt.")
    parser.add_argument("--max_tokens",
                        type=int,
                        default=500,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.7,
                        help="Temperature for the model.")
    parser.add_argument("--base_url",
                        type=str,
                        required=True,
                        help="Base URL of the API.")
    parser.add_argument("--model",
                        type=str,
                        default="llama4:latest",
                        help="Model to use for inference.")
    args = parser.parse_args()

    base_url = args.base_url

    # Determine the base URL based on the model type.
    # For ollama models, use the ollama endpoint.
    # For OpenAI models, use the completions endpoint.
    if args.model in ["o4-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"]:
        base_url += "api/chat/completions"
    # For other models, use the ollama endpoint. Note that this has only
    # been tested with llama4:latest (llama scout) so far.
    else:
        base_url += "ollama/api/chat"

    run_inference(prompt_file=args.prompt_file,
                  output_path=args.output_path,
                  api_key=args.api_key,
                  base_url=base_url,
                  repeat=args.repeat,
                  output_file=args.output_file,
                  max_tokens=args.max_tokens,
                  temperature=args.temperature,
                  model=args.model)

