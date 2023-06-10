import requests

# Replace with your OpenAI API key
API_KEY = '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'


def call_openai_api(prompt):
    url = 'https: // api.openai.com/v1/chat/completions'

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    data = {
        'prompt': prompt,
        'max_tokens': 150,
        'n': 1,
        'stop': None,
        'temperature': 0.8,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['text'].strip()
    else:
        print(f'Error: {response.status_code}')
        return None


def main():
    while True:
        user_input = input('User: ')
        if user_input.lower() == 'quit':
            break

        prompt = f'You are ChatGPT, a large language model trained by OpenAI. A user asks: {user_input}. Your response:'
        response = call_openai_api(prompt)
        if response:
            print(f'ChatGPT: {response}')
        else:
            print('Error: Unable to get a response from the API.')


if __name__ == '__main__':
    main()
