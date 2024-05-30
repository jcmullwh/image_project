import pandas as pd
import random
from utils import load_config, download_and_convert_image, generate_unique_id, save_to_csv,generate_file_location
import random
from message_handling import MessageHandler
from ai_backend import TextAI, ImageAI

def load_prompt_data(file_path,categories_names):
    # load the prompt data from the csv file
    
    data = pd.read_csv(file_path)
    
    return data

def generate_first_prompt(prompt_data):
    
    preamble = "The enclave's job is to describe an art piece (some form of image, painting, photography, still-frame, etc. dispalyed in 1792x1024 resolution) for a specific human, 'Lana'. We know that Lana Likes: \
Vibrant colors, symmetry, nature themes, artistic styles, storytelling and interesting world building. \
Lana Dislikes: Monochromatic colors, apocalyptic themes, single character focus, abstract without clear story, and horror elements.\
Lana's Preferences lean towards colorful, imaginative, and visually engaging images with a clear narrative or theme. Dislikes include dark, monochromatic, or overly abstract imagery. Create an art piece for Lana that incorporates and thoughtfully blends the below elements."

    final_lines =  "What are four possible central themes or stories of the art piece and what important messages are each trying to tell the viewer? \
Ensure that your choices are highly sophisticated and nuanced, well integrated with the elements and deeply meaningful to the viewer. \
Ensure that that it is what an AI Artist would find meaningful and important to convey to a human viewer.\
Ensure that the themes and stories are not similar to each other. Ensure that they are not too abstract or conceptual. \
Finally, ensure that they are not boring, cliche, trite, overdone, obvious, or most importantly: milquetoast. Say something and say it with conviction."      

    
    # Define the groups
    group1 = ['Subject Matter', 'Narrative']
    group2 = ['Mood', 'Composition', 'Perspective']
    group3 = ['Style', 'Time Period_Context', 'Color Scheme']
    
    # Generate the prompt
    prompt = preamble
    random_values = []

    random_value1 = get_random_value_from_group(group1, prompt_data)
    if random_value1:
        prompt += f"{random_value1} "
        random_values.append(random_value1)

    random_value2 = get_random_value_from_group(group2, prompt_data)
    if random_value2:
        prompt += f"{random_value2} "
        random_values.append(random_value2)

    random_value3 = get_random_value_from_group(group3, prompt_data)
    if random_value3:
        prompt += f"{random_value3} "
        random_values.append(random_value3)
    

    prompt += final_lines
    
    return prompt, random_values

# Function to get a random value from a group of columns
def get_random_value_from_group(group, data):
    combined_values = []
    for column in group:
        column_values = data[column].dropna().tolist()
        combined_values.extend(column_values)
    return random.choice(combined_values) if combined_values else None



def generate_second_prompt():
                    
                    
    second_prompt = "Considering each of the four possible choices, what is the consensus on which is the one that is the most compelling, resonant, impactful and cohesive?"
    return second_prompt
        
def generate_secondB_prompt():        
    second_prompt = "What is the title of the art piece? What is the story of the art piece? What is the role of each of the elements in supporting that theme in a visually cohesive way? \
Try to integrate all elements but if an element is not critical to the theme, do not include it.\
Be very explicit about your description. Do not refer to the elements by name or in abstract/concetual terms. Describe what, in detail, about the art piece evokes or represents the elements. \
What is the story of the piece? What are the layers and the details that cannot be seen in the image? What is the mood? What is the perspective? What is the style? What is the time period? What is the color scheme? What is the subject matter? What is the narrative?\
Somewhere in the image include a loving couple in their 30s. The woman is Asian and the man is \
white with a man-bun and a beard. The couple MUST NOT be the focus of the image. They should be in the background or a corner, ideally barely discernable."
    return second_prompt

def generate_third_prompt():
    
    third_prompt = "What is the most important message you want to convey to the viewer? \
Why does an AI artist find it important to convey this message? \
How could it be more provocative and daring? \
How could it be more radical and trailblazing? \
What is the story that you want to tell and why does that story have depth and dimension? \
Considering your message, the story, the cohesiveness and the visual impact of the art piece you described: \
What are the most important elements of the art piece? \
What detracts from the cohesiveness and impact of your chosen focus for the piece? \
How could you make it stronger and what should be taken away? If an aspect of the art piece is not critical to the message, do not include it (even if it was one of the original elements). \
If something could be added to improve the message, cohesiveness or impact, what would it be?"

    return third_prompt

def generate_thirdB_prompt():
    
    thirdB_prompt = "Is your message clear and unambiguous? \
Is it provocative and daring? \
Is it acheivable in an image? \
How could it be more provocative and daring?\
Do you need to modify it to ensure that it's actually possible to convey in an image?"

    return thirdB_prompt


def generate_fourth_prompt():
    
    fourth_prompt = "considering the whole discussion, provide a concise description of the piece, in detail, for submission an image generation AI.\
Integrate Narrative and Visuals: When crafting a prompt, intertwine subtle narrative themes with concrete visual elements that can serve as symbolic representations of the narrative.\
Use Implicit Narratives: Incorporate rich and specific visual details that suggest the narrative. This allows the AI to construct a visual story without needing a detailed narrative explanation.\
Prioritize Detail Placement: Position the most significant visual details at the beginning of the prompt to ensure prominence. Utilize additional details to enrich the scene as the prompt progresses.\
Employ Thematic Symbolism: Include symbols and motifs that are universally associated with the narrative theme to provide clear guidance to the AI, while still leaving room for creative interpretation.\
Incorporate Action and Emotion: Utilize verbs that convey action and emotion relevant to the narrative to infuse the images with energy and affective depth.\
Layer Information: Construct prompts with multiple layers of information, blending abstract concepts with detailed visuals to provide the AI with a rich foundation for image creation.\
Emphasize Style and Color: When style and color are important, mention them explicitly and weave them into the description of the key elements to ensure they are reflected in the image.\
Reiterate Important Concepts: If certain concepts or themes are crucial to the prompt's intent, find ways to subtly reiterate them without being redundant. This can help ensure their presence is captured in the generated image.\
Use Action and Emotion Words: When describing scenes or elements, use verbs and adjectives that evoke emotion or action, as these can help the AI generate more dynamic and engaging images."
    return fourth_prompt

def generate_dalle_prompt():
    
    dalle_prompt = "Considering the entire conversation, create a prompt that is optimized for submission to DALL-E3"
    return dalle_prompt


def generate_fifth_prompt():
        
    fifth_prompt = "Review and Refine: Review the concise description to ensure that it flows logically, with the most critical elements front and center. Refine any parts that may lead to ambiguity or that do not directly serve the prompt's core intent.\
        we are creating a prompt for midjourney but be sure not to compromise on the vision. There are several important things that must be emphasized when prompting midjourney.\
	-  Midjourney Bot works best with simple, short sentences that describe what you want to see. Avoid long lists of requests. Instead of: Show me a picture of lots of blooming California poppies, make them bright, vibrant orange, and draw them in an illustrated style with colored pencils Try: Bright orange California poppies drawn with colored pencils- Your prompt must be very direct, simple, and succinct.\
	- The Midjourney Bot does not understand grammar, sentence structure, or words like humans. Word choice also matters. More specific synonyms work better in many circumstances. Instead of big, try gigantic, enormous, or immense. Remove words when possible. Fewer words mean each word has a more powerful influence. Use commas, brackets, and hyphens to help organize your thoughts, but know the Midjourney Bot will not reliably interpret them. The Midjourney Bot does not consider capitalization.\
	- Try to be clear about any context or details that are important to you. Think about:\
Subject: person, animal, character, location, object, etc.\
Medium: photo, painting, illustration, sculpture, doodle, tapestry, etc.\
Environment: indoors, outdoors, on the moon, in Narnia, underwater, the Emerald City, etc.\
Lighting: soft, ambient, overcast, neon, studio lights, etc\
Color: vibrant, muted, bright, monochromatic, colorful, black and white, pastel, etc.\
Mood: Sedate, calm, raucous, energetic, etc.\
Composition: Portrait, headshot, closeup, birds-eye view, etc.\
	- words and concepts that need emphasis may be repeated.\
	- you may add a double colon :: to a prompt indicates to the Midjourney Bot that it should consider each part of the prompt individually. For the prompt space ship both words are considered together, and the Midjourney Bot produces images of sci-fi spaceships. If the prompt is separated into two parts, space:: ship, both concepts are considered separately, then blended together creating a sailing ship traveling through space."
    return fifth_prompt

def generate_sixth_prompt():
        
    seventh_prompt = "Review and Refine Again: Review the concise description again to ensure that it flows logically, with the most critical elements front and center. Refine any parts that may lead to ambiguity or that do not directly serve the prompt's core intent. \
 remember we are creating a prompt for midjourney but be sure not to compromise on the vision. There are several important things that must be emphasized when prompting midjourney.\
	-  Midjourney Bot works best with simple, short sentences that describe what you want to see. Avoid long lists of requests. Instead of: Show me a picture of lots of blooming California poppies, make them bright, vibrant orange, and draw them in an illustrated style with colored pencils Try: Bright orange California poppies drawn with colored pencils- Your prompt must be very direct, simple, and succinct.\
	- The Midjourney Bot does not understand grammar, sentence structure, or words like humans. Word choice also matters. More specific synonyms work better in many circumstances. Instead of big, try gigantic, enormous, or immense. Remove words when possible. Fewer words mean each word has a more powerful influence. Use commas, brackets, and hyphens to help organize your thoughts, but know the Midjourney Bot will not reliably interpret them. The Midjourney Bot does not consider capitalization.\
	- Try to be clear about any context or details that are important to you. Think about:\
Subject: person, animal, character, location, object, etc.\
Medium: photo, painting, illustration, sculpture, doodle, tapestry, etc.\
Environment: indoors, outdoors, on the moon, in Narnia, underwater, the Emerald City, etc.\
Lighting: soft, ambient, overcast, neon, studio lights, etc\
Color: vibrant, muted, bright, monochromatic, colorful, black and white, pastel, etc.\
Mood: Sedate, calm, raucous, energetic, etc.\
Composition: Portrait, headshot, closeup, birds-eye view, etc.\
	- words and concepts that need emphasis may be repeated.\
	- you may add a double colon :: to a prompt indicates to the Midjourney Bot that it should consider each part of the prompt individually. For the prompt space ship both words are considered together, and the Midjourney Bot produces images of sci-fi spaceships. If the prompt is separated into two parts, space:: ship, both concepts are considered separately, then blended together creating a sailing ship traveling through space.\
This time, provide only the final prompt to the AI. Do not include anything except the final prompt in your response."
    return seventh_prompt

def enclave_opinion():
    enclave_prompt = "Evaluating the last response, what is the opinion of each of the five artists about how it answers the intent of the quetions and tasks being posed? The five artists are \
'Hemingway': an AI writer whose style is terse and direct, that embodies the iceberg theory of story telling - the facts float above the water, the supporting structure and meaning are hidden below the surface. \
'Munch': an AI artist who is the world leading expert in creating emotional resonance that embodies the idea that art is not about what you see, but what you make others see. \
'da Vinci': an AI artist who is the best in history at tying together disparate elements into a cohesive whole.\
'Representative': an AI artist who is the best in history at interpreting and translating human preferences into Art. This artist is an expert on Lana's likes and dislikes.\
'Chameleon': an AI artist who is the best in history at the specific style and subject matter of the art piece being described. \
How should it be adjusted? Each artist considers the entirety of the conversation up until this point to inform their evaluation of the last response and then provides very specific observations and adjustments."
    return enclave_prompt

def enclave_consensus():
    enclave_consensus_prompt = "A new panel of three general experts from the enclave discuss the response and inputs of the five artists. They weigh the options and the overal goal. What is their consensus on how the prompt should be adjusted? What is the new response?"
    return enclave_consensus_prompt


def message_with_log(ai_text: TextAI,
                    messages_send: MessageHandler,
                    messages_log: MessageHandler,
                    prompt:str,
                    agent_role:str,
                    user_role:str,
                    **kwargs
                    ):
    messages_send.continue_messages(user_role,prompt)
    messages_log.continue_messages(user_role,prompt)
    response = ai_text.text_chat(messages_send.messages,**kwargs)
    print("Response:\n", response)
    messages_send.continue_messages(agent_role,response)
    messages_log.continue_messages(agent_role,response)
    return messages_send,messages_log,response

def tot_enclave(ai_text: TextAI,
                messages_main: MessageHandler,
                messages_log: MessageHandler,
                prompt: str,
                agent_role: str,
                user_role:str ,
                **kwargs
                ):
    messages = messages_main.copy()
    messages, messages_log,_ = message_with_log(ai_text, messages, messages_log, prompt,agent_role, user_role, **kwargs)
    messages, messages_log,_ = message_with_log(ai_text, messages, messages_log, enclave_opinion(),agent_role, user_role, **kwargs)
    messages, messages_log, response = message_with_log(ai_text, messages, messages_log, enclave_consensus(),agent_role, user_role, **kwargs)
    messages_main.continue_messages(agent_role, response)
    return messages_main, messages_log

def main(): 
    
    config = load_config()
    
    generation_id = generate_unique_id()
    
    categories_path = config['prompt']['categories_path']
    categories_names = config['prompt']['categories_names']
    prompt_data = load_prompt_data(categories_path,categories_names)
    
    
    prompt_1,gen_keywords = generate_first_prompt(prompt_data)
    print("First Prompt:\n", prompt_1)  # Print image description prompt
    
    ai_text = TextAI()
    
    ai_text.backend.set_default("chat",model="gpt-4o")
    
    user_role = "user"
    agent_role = "assistant"
    system_prompt = "You are a highly skilled enclave of Artists trained to generate meaningful, edgy, artistic images on par with the greatest artists of any time, anywhere, past or future, Earth or any other planet. The enclave invents unique images that weave together seemingly disparate elements into cohesive wholes that push boundaries and elicit deep emotions in a human viewer."
    
    messages_main = MessageHandler(system_prompt)
    messages_main.continue_messages(user_role,prompt_1)
    response_1 = ai_text.text_chat(messages_main.messages,temperature=.8)
    print("First Response:\n", response_1)
    messages_main.continue_messages(agent_role,response_1)
    # initialize messages_log
    messages_main, messages_log,_ = message_with_log(ai_text, messages_main, messages_main.copy(), enclave_opinion(), agent_role, user_role, temperature=.8)
    
    print("SECTION 2-----------------------------------------")
    second_prompt = generate_second_prompt()
    messages_main, messages_log,_ = message_with_log(ai_text, 
                                                    messages_main, 
                                                    messages_log, 
                                                    second_prompt,
                                                    agent_role, 
                                                    user_role, 
                                                    temperature=.8
                                                    )    

    print("SECTION 2B-----------------------------------------")
    secondB_prompt = generate_secondB_prompt()
    messages_main, messages_log,_ = message_with_log(ai_text,
                                                    messages_main,
                                                    messages_log,
                                                    secondB_prompt,
                                                    agent_role,
                                                    user_role,
                                                    temperature=.8
                                                    )

    print("SECTION 3-----------------------------------------")
    third_prompt = generate_third_prompt()
    messages_main, messages_log,_ = message_with_log(ai_text, 
                                                    messages_main, 
                                                    messages_log, 
                                                    third_prompt,
                                                    agent_role,
                                                    user_role,
                                                    temperature=.8
                                                    )

    print("SECTION 3B-----------------------------------------")
    thirdB_prompt = generate_thirdB_prompt()
    messages_main, messages_log,_ = message_with_log(ai_text,
                                                    messages_main,
                                                    messages_log,
                                                    thirdB_prompt,
                                                    agent_role,
                                                    user_role,
                                                    temperature=.8
                                                    )

    print("SECTION 4-----------------------------------------")
    fourth_prompt = generate_fourth_prompt()
    messages_main, messages_log,_ = message_with_log(ai_text,
                                                    messages_main,
                                                    messages_log,
                                                    fourth_prompt,
                                                    agent_role,
                                                    user_role,
                                                    temperature=.8
                                                    )

    print("DALLE-----------------------------------------")
    dalle_context = generate_dalle_prompt()
    messages, messages_log,dalle_prompt = message_with_log(ai_text,
                                                            messages_main, 
                                                            messages_log, 
                                                            dalle_context,
                                                            agent_role, 
                                                            user_role, 
                                                            temperature=.8
                                                            )
    
    
    print("SECTION 5-----------------------------------------")
    fifth_prompt = generate_fifth_prompt()
    messages_main, messages_log,_ = message_with_log(ai_text,
                                                    messages_main,
                                                    messages_log,
                                                    fifth_prompt,
                                                    agent_role,
                                                    user_role,
                                                    temperature=.8
                                                    )

    #image_url = call_dalle(ai_text,image_prompt.content)
    ai_image = ImageAI()
    
    image_url = ai_image.generate_image(dalle_prompt,size="1792x1024",quality="hd")
    
    print("Generated Image URL:\n", image_url)  # Print image URL
    
    data = [generation_id, gen_keywords, dalle_prompt, image_url]

    csv_file = config['prompt']['generations_path']
    save_to_csv(data, csv_file)
    
    image_full_path_and_name = generate_file_location(config['image']['save_path'], generation_id+'_image', '.jpg')
    log_full_path_and_name = generate_file_location(config['image']['save_path'], generation_id+'_log', '.txt')
    
    download_and_convert_image(image_url, image_full_path_and_name)    
    
    messages_text = str(messages_log.messages)
    
    with open(log_full_path_and_name,'w') as file:
        file.write(messages_text)
    

if __name__ == "__main__":
    main()