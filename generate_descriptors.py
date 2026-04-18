import os
import openai
import json

import itertools

from descriptor_strings import stringtolist

openai.api_key = os.environ["OPENAI_API_KEY"]


def generate_prompt1(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: Describe what an animal lemur looks like, list 7 pieces?
A:
- has four-limbed primate
- is black, grey, white, brown, or red-brown
- owns wet and hairless nose with curved nostrils
- exhibit long tail
- exhibit large eyes
- appear as furry bodies
- has clawed hands and feet

Q: Describe what an appliance television looks like, list 6 pieces?
A:
- is a electronic device
- is black or grey
- has a large, rectangular screen
- owns a stand or mount to support the screen
- has one or more speakers
- attach a power cord

Q: Describe what an object {category_name} looks like, list 10 pieces?
A:
-
"""

def generate_prompt2(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: Visually describe a lemur, a type of animal, list 7 pieces?
A: Here is 7 pieces of features:
- has four-limbed primate
- is black, grey, white, brown, or red-brown
- owns wet and hairless nose with curved nostrils
- exhibit long tail
- exhibit large eyes
- appear as furry bodies
- has clawed hands and feet

Q: Visually describe a television, a type of appliance, list 6 pieces?
A: Here is 6 pieces of features:
- is a electronic device
- is black or grey
- has a large, rectangular screen
- owns a stand or mount to support the screen
- has one or more speakers
- attach a power cord

Q: Visually describe a {category_name}, a type of object,, list 10 pieces?
A: Here is 10 pieces of features:
-
"""

# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))
        
def obtain_descriptors_and_save(filename, class_list):
    responses = {}
    descriptors = {}
    
    ###################################
    
    prompts = [generate_prompt1(category.replace('_', ' ')) for category in class_list]
    
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    responses = [openai.Completion.create(model="text-davinci-003",
                                            prompt=prompt_partition,
                                            temperature=0.5,
                                            max_tokens=100,
                                            ) for prompt_partition in partition(prompts, 20)]
    response_texts = [r["text"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]

    # truncate
    min_length = min([len(descriptor) for descriptor in descriptors_list])
    descriptors_list1 = [descriptor[:min(min_length, 20)] for descriptor in descriptors_list]

    ###################################

    prompts = [generate_prompt2(category.replace('_', ' ')) for category in class_list]
    
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    responses = [openai.Completion.create(model="text-davinci-003",
                                            prompt=prompt_partition,
                                            temperature=0.5,
                                            max_tokens=100,
                                            ) for prompt_partition in partition(prompts, 20)]
    response_texts = [r["text"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]

    # truncate
    min_length = min([len(descriptor) for descriptor in descriptors_list])
    descriptors_list2 = [descriptor[:min(min_length, 20)] for descriptor in descriptors_list]

    ###################################

    descriptors_list3 = []
    for l1, l2 in zip(descriptors_list1, descriptors_list2):
        descriptors_list3.append(l1 + l2)

    ###################################

    for idx in range(len(class_list)):
        classname = class_list[idx]
        descriptors_list3[idx] = [classname + " which " + attr for attr in descriptors_list3[idx]]
    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors_list3, fp)
    

obtain_descriptors_and_save('example', ['llama', 'skunk', 'badger', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'baboon', 'giant panda', 'eel', 'clownfish', 'pufferfish', 'accordion', 'ambulance', 'assault rifle', 'backpack', 'barn', 'wheelbarrow', 'basketball', 'bathtub', 'lighthouse', 'beer glass', 'binoculars', 'birdhouse', 'bow tie', 'broom', 'bucket', 'cauldron', 'candle', 'cannon', 'canoe', 'carousel', 'castle', 'mobile phone', 'cowboy hat', 'electric guitar', 'fire truck', 'flute', 'gas mask or respirator', 'grand piano', 'guillotine', 'hammer', 'harmonica', 'harp', 'hatchet', 'jeep', 'joystick', 'lab coat', 'lawn mower', 'lipstick', 'mailbox', 'missile', 'mitten', 'parachute', 'pickup truck', 'pirate ship', 'revolver', 'rugby ball', 'sandal', 'saxophone', 'school bus', 'schooner', 'shield', 'soccer ball', 'space shuttle', 'spider web', 'steam locomotive', 'scarf', 'submarine', 'tank', 'tennis ball', 'tractor', 'trombone', 'vase', 'violin', 'military aircraft', 'wine bottle', 'ice cream', 'bagel', 'pretzel', 'cheeseburger', 'hot dog', 'cabbage', 'broccoli', 'cucumber', 'bell pepper', 'mushroom', 'Granny Smith apple', 'strawberry', 'lemon', 'pineapple', 'banana', 'pomegranate', 'pizza', 'burrito', 'espresso', 'volcano', 'baseball player', 'scuba diver', 'acorn'])