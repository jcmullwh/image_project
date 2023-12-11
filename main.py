import pandas as pd
import random
from utils import load_config, download_and_convert_image, generate_unique_id, save_to_csv
import random
from openai import OpenAI
from openai_api import call_gpt4, call_dalle
import uuid

def load_prompt_data(file_path,categories_names):
    # Adjust the path if your main.py is located in a different folder
    
    data = pd.read_csv(file_path)
    
    return data

def generate_description_prompt(prompt_data):
    
    preamble = 'Describe an art piece that incorporates and thoughtfully blends the below elements.\
                    What is the central theme of the image and why does it inherently require each of these elements? \
                    What is the role of each of the elements in supporting that theme in a visually cohesive way?' 
                                        
    couple_prompt = 'Somewhere in the image include a loving couple in their 30s. The woman is Asian and the man is \
        white with a man-bun and a beard. The couple should not be the focus of the image.'
        
    final_lines = 'It is important that you detail an image which the viewer might describe each of these elements. \
                It must all make sense together. Why would it makes sense for all of these things to come together and form not just an image but high art?"'
    
    prompt = preamble
    random_values = []
    
    for column in prompt_data.columns:
        column_values = prompt_data[column].dropna().tolist()
        if column_values:
            random_value = random.choice(column_values)
            prompt += f" Why will the viewer experience {random_value} as an integral part of the image?"
            random_values.append(random_value)
    
    prompt += couple_prompt
    prompt += final_lines
    
    return prompt, random_values

def main():
    
    config = load_config()
    
    generation_id = generate_unique_id()
    
    categories_path = config['prompt']['categories_path']
    categories_names = config['prompt']['categories_names']
    prompt_data = load_prompt_data(categories_path,categories_names)
    
    
    image_description_prompt,gen_keywords = generate_description_prompt(prompt_data)
    print("Image Description Prompt:\n", image_description_prompt)  # Print image description prompt

    
    client = OpenAI(api_key=config['openai']['key'])
    image_generation_prompt = call_gpt4(client,image_description_prompt)
    print("Image Generation Prompt:\n", image_generation_prompt)  # Print image generation prompt

    image_url = call_dalle(client,image_generation_prompt)
    print("Generated Image URL:\n", image_url)  # Print image URL
    
    data = [generation_id, gen_keywords,image_description_prompt, image_generation_prompt, image_url]

    csv_file = config['prompt']['generations_path']
    save_to_csv(data, csv_file)
    
    download_and_convert_image(image_url, generation_id, config['image']['save_path'])    
    

if __name__ == "__main__":
    main()