from openai import OpenAI
import requests

def call_gpt4(client, prompt,model="gpt-4",temperature=0.2):
    
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly skilled AI trained to generate meaningful, artistic images on par with the greatest artists of any time, past or future. You invent unique images that weave together seemingly disparate elements into cohesive wholes that elicit deep emotions in human viewers.Your goal is to create art that will be hung in the Louvre, the Met, and the Tate."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")        

def call_dalle(client,image_prompt):

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            size="1792x1024",
            quality="hd",
            n=1,
        )
        
        image_url = response.data[0].url
        return image_url
    
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")   
