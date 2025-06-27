import cv2
import base64
import json
import os
import ast
import numpy as np
from datetime import datetime
import openai
from openai import OpenAI
import time
from tqdm import tqdm
import re
# Configuration
BOUNDING_BOX_DESCRIPTION_PATH = "./bbox_descriptions"
if not os.path.exists(BOUNDING_BOX_DESCRIPTION_PATH):
    os.makedirs(BOUNDING_BOX_DESCRIPTION_PATH)
parent_path = "/root/paddlejob/workspace/env_run/text2image/t2i_data"
# System prompt for combined descriptions

api_key_ls = [
    "sk---fenghaoxuan---A56OGa9I8w/7pIVTixJGUA==",
    "sk---v_lvxu---t3vmU7XgWAyFppXdmzZWPw==", 
    "sk---xumingyang02---UBrVCrYoKlriamPUBuajFQ==",
    "sk---xiachunlong---rsnPjBkhu7cnvZ3hDbPXnA==",
    "sk---v_zhangyin05---IO+5mjbExXjFBd0AwX6Fjw==",
    "sk---wangshuo36---2jIg+BlCdUnybJHEPl58sQ==",
    "sk---zhangyujian02---8gjJfLJZ5IXLDls+Olh3Ug==",
    "sk---lvfeng02---Fc2U9KliFHeQyEu+YSLsIQ==",
    "sk---hufeiyu---uP8bUlP6skVRjfFqfsQuqw==",
    "sk---yiguangqi---/EtQJhilTO5TXXGRkr/7aQ==",
    "sk---wanglingyan02---77Z8PDLEkd+niLj9pvW+sw==",
    "sk---v_hebo03---bcIyi2R3DcHjuSCmpmUh/A==",
    "sk---v_sunchuang---CKOGjEmpGy/bNOdjgZFR7Q==",
    "sk---wangchonghuinan---Ib2ccPs1BNvcWEBbyykfHw==",
    "sk---mujunxian---p53Do0F8QacToWiFC8719w==",
    "sk---wangzhao04---3kbKawb8icytuFNbDhR5qg=="
]

model_priority = [
    'gpt-4o-2024-08-06',
    'gpt-4-turbo',
    'gpt-4-vision',
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
]
current_api_key_index = 0
current_model_index = 0
max_retries_per_model = 2
max_retries_per_key = 3  # Number of retries before switching keys
def get_current_api_key():
    """Get the current API key and rotate if needed"""
    global current_api_key_index
    return api_key_ls[current_api_key_index]

def get_current_model():
    """Get the current model to try"""
    global current_model_index
    if current_model_index < len(model_priority):
        return model_priority[current_model_index]
    return None

def rotate_model():
    """Switch to the next model in the list"""
    global current_model_index
    current_model_index += 1
    if current_model_index >= len(model_priority):
        current_model_index = 0
        return False  # Indicates we've tried all models
    return True  
def rotate_api_key():
    """Switch to the next API key in the list"""
    global current_api_key_index
    current_api_key_index = (current_api_key_index + 1) % len(api_key_ls)
    print(f"Rotating to API key index {current_api_key_index}")
    openai.api_key = get_current_api_key()

bbox_combined_system = '''
### Task:
You are an expert at generating multiple description levels for objects detected in images. Given a cropped object image, create three description types:
1. Tag (1-3 words)
2. Short description (1 sentence, 10-15 words)
3. Detailed description (2-3 sentences)

### Guidelines:
For TAG:
- Be extremely concise (1-3 words max)
- Use common terminology
- Focus on primary object identity

For SHORT DESCRIPTION:
- Keep to one simple sentence
- Include key attributes like color, size, orientation
- Be factual based on visible features

For DETAILED DESCRIPTION:
- Be thorough but concise (2-3 sentences)
- Include:
  - Object type and key attributes
  - Visible condition/state
  - Notable features
  - Context inferred from visible parts

### Output Format:
Return JSON with all three description types:
{
  "tag": "your_tag_here",
  "short_description": "your_short_description",
  "detailed_description": "your_detailed_description"
}
'''

def process_input_line(line):
    """Parse input line into components"""
    parts = line.strip().split()
    if len(parts) != 6:
        raise ValueError(f"Invalid input format. Expected 6 parts, got {len(parts)}")
    
    return {
        "class": parts[0],
        "image_path": parts[1],
        "bbox": [int(x) for x in parts[2:6]]  # x1,y1,x2,y2
    }
def parse_model_response(response_text):
    # 清理 Markdown 代码块（如果存在）
    response_text = response_text.strip()
    if response_text.startswith('```json'):
        response_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.DOTALL)
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}\nRaw response: {response_text}")
        return {
            "tag": "error",
            "short_description": "Failed to parse description",
            "detailed_description": "The model returned an invalid format."
        }
def crop_and_encode_image(image_path, bbox, min_size=64):
    """Crop object based on bbox and encode as base64"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    x1, y1, x2, y2 = bbox
    # 随机扩散
    x1 = x1 - min(10, bbox[0]-0)
    y1 = y1 - min(10, bbox[1]-0)
    x2 = x2 + min(10, img.shape[1]-bbox[2])
    y2 = y2 + min(10, img.shape[0]-bbox[3])
    
    cropped_img = img[y1:y2, x1:x2]
    crop_h, crop_w = cropped_img.shape[:2]
        
    if crop_h < min_size or crop_w < min_size:
        print(f"Skipped: Cropped size too small ({crop_w}x{crop_h} < {min_size}x{min_size})")
        return None
    # Resize if too large (to save tokens)
    max_dim = 512
    h, w = cropped_img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        cropped_img = cv2.resize(cropped_img, (int(w*scale), int(h*scale)))
    
    _, buffer = cv2.imencode(".png", cropped_img)
    return base64.b64encode(buffer).decode("utf-8")

def generate_combined_descriptions(base64_image, default_model='gpt-4o-2024-08-06'):
    """Generate all three description types with model and key rotation"""
    global current_api_key_index, current_model_index
    
    for key_attempt in range(len(api_key_ls)):
        # Try up to max_models_per_key models with this API key
        for model_attempt in range(max_retries_per_model):
            current_model = get_current_model()
            if not current_model:
                print("No more models to try with current API key")
                break
                
            print(f"Trying model: {current_model} with API key {current_api_key_index}")
            
            for attempt in range(max_retries_per_model):
                try:
                    completion = openai.chat.completions.create(
                        model=current_model,
                        temperature=0,
                        top_p=0.1,
                        messages=[
                            {"role": "system", "content": bbox_combined_system},
                            {"role": "user", "content": [
                                {"type": "text", "text": "Please describe this object at three levels of detail as specified."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                            ]}
                        ]
                    )
                    return parse_model_response(completion.choices[0].message.content)
                
                except (openai.RateLimitError, openai.APIError) as e:
                    print(f"Attempt {attempt + 1} failed with model {current_model}: {str(e)}")
                    if attempt == max_retries_per_model - 1:
                        # Try next model
                        if not rotate_model():
                            # No more models to try with this key
                            break
                        continue
                    time.sleep(2 ** attempt)  # Exponential backoff
                
                except Exception as e:
                    print(f"Error with model {current_model}: {str(e)}")
                    if "quota" in str(e).lower() or "limit" in str(e).lower():
                        # Quota exceeded, try next key
                        break
                    return {
                        "tag": "unknown",
                        "short_description": "Unable to generate description",
                        "detailed_description": "Failed to generate description due to processing error"
                    }
        
        # If we get here, all models failed with this key - rotate key
        rotate_api_key()
    
    # If all keys exhausted
    raise Exception("All API keys and models have been exhausted")

def process_single_object(obj_data, version='gpt-4o-2024-08-06'):
    """Process a single object from input line"""
    try:
        # Crop and encode the object
        base64_img = crop_and_encode_image(os.path.join(parent_path,obj_data["image_path"]), obj_data["bbox"])
        if base64_img is None:
            return None
        # Get all descriptions in one call
        descriptions = generate_combined_descriptions(base64_img, version)
        
        return {
            "class": obj_data["class"],
            "image_path": obj_data["image_path"],
            "bbox": obj_data["bbox"],
            **descriptions
        }
    except Exception as e:
        print(f"Error processing object: {str(e)}")
        return {
            "class": obj_data["class"],
            "image_path": obj_data["image_path"],
            "bbox": obj_data["bbox"],
            "tag": "error",
            "short_description": "Processing failed",
            "detailed_description": f"Error: {str(e)}"
        }

def process_input_file(input_path, output_dir=BOUNDING_BOX_DESCRIPTION_PATH, version='gpt-4o-2024-08-06'):
    """Process an input file containing multiple objects"""
    results = []
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="Processing objects"):
        try:
            obj_data = process_input_line(line)
            result = process_single_object(obj_data, version)
            if result is None:
                continue
            results.append(result)
        except Exception as e:
            print(f"Skipping malformed line: {line.strip()}. Error: {str(e)}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"descriptions_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump({"results": results}, f, indent=2)
    
    print(f"Saved descriptions for {len(results)} objects to {output_path}")
    return output_path

# Example usage
if __name__ == '__main__':
    # Initialize OpenAI client
    openai.base_url = "http://llms-se.baidu-int.com:8200"
    openai.default_headers = {"x-foo": "true"}
    openai.api_key = get_current_api_key()  # Initialize with first key
    
    # Process input file
    input_file = "/root/paddlejob/workspace/env_run/text2image/t2i_data/fokai01/train/train.txt"
    output_file = process_input_file(input_file)
    
    print(f"Processing complete. Results saved to {output_file}")
