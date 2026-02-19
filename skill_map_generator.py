import json
import os
import boto3
from pymongo import MongoClient
from dotenv import load_dotenv
import time
load_dotenv()

bedrock_client = boto3.client(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    service_name="bedrock-runtime",
    region_name="ap-south-1"
)

mongo_url = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(mongo_url)
db = mongo_client["chatbot"]
users_collection = db['users']

def skill_split(user_id: str, thread_id: str, skill: str):
    """Generate skill map and store in MongoDB."""
    print(f"skill_split: Starting for user={user_id}, thread={thread_id}, skill={skill}")
    
    # Set status to building
    users_collection.update_one(
        {"user_id": user_id},
        {
            "$set": {
                f"threads.{thread_id}.skill_map_status": "building"
            }
        },
        upsert=True
    )
    
    try:
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Chat title (3-5 words)"},
                "skill_map": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "progress": {"type": "integer", "default": 0},
                        "depth": {"type": "integer", "enum": [0, 1, 2, 3]},
                        "contents": {"type": "object"}
                    },
                    "required": ["title", "progress", "depth", "contents"]
                }
            },
            "required": ["title", "skill_map"]
        }
        
        instructions = """You are an AI system tasked with generating hierarchical skill maps to structure learning paths.

1. Analyze the skill provided within <input> tags.
2. Return a JSON response that complies with the provided schema.
3. Do NOT wrap response in markdown code blocks.
4. Output ONLY raw JSON.
---
DEPTH DECISION RULES:
- Depth = 0: The bottom most level of the skill map. This is the most specific level of the skill map. This is the level where the user can learn the most specific topics. 
  Structure: depth 0 root → contents (3-5 subskills with ONLY {{"progress": 0}})

- Depth = 1: Extremely specific topics (4+ words, single action/feature)
  Structure: depth 1 root → depth 0 leaves (with depth and progress) → contents (3-5 subskills with ONLY {{"progress": 0}})
  
- Depth = 2: Single technology/domain (1-3 words)
  Structure: depth 2 root → depth 1 topics (with depth and progress) → depth 0 skills (with depth and progress) → contents (3-5 subskills with ONLY {{"progress": 0}})
  
- Depth = 3: Multiple domains/techs (contains "and", "for", "with")
  Structure: depth 3 root → depth 2 domains (with depth and progress) → depth 1 topics (with depth and progress) → depth 0 skills (with depth and progress) → contents (3-5 subskills with ONLY {{"progress": 0}})
---
HOW TO GENERATE THE SKILL MAP:
1. Understand the depth of the user's skill.
2. Segregrate into well defined contents and subtopics. 
3. Map it out clearly and then understand the depth of the skill map. 
4. You must always teach the other user all the neccessary topics for a given skill map. Use your 
extensive knowledge to decide this first and then generate the skill map. You cannot skip any of the 
vital topics for a given skill map. Do not cram everything into 1 depth heading to make the skill map precise or anything. If a split of that topic is better for the user, then please do split it following the structure rules mentioned above and the example below. 
5. The headings you chose for any of the depths, must be short and concise. Do not make them too long. 
---
CRITICAL STRUCTURE RULES:
- EVERY node MUST have "progress" and "depth" keys EXCEPT the final leaf subskills
- Parent depth MUST be EXACTLY 1 greater than child depth (depth 2 → depth 1 → depth 0)
- Final leaf subskills (inside depth=0 contents) ONLY have {{"progress": 0}} - NO depth key
- Depth=0 nodes MUST have 3-5 subskills in their contents dictionary
- All names must be unique and relevant
- NEVER skip intermediate depth levels - if root is depth=2, you MUST have depth=1 nodes, then depth=0 nodes
- The skill map and all its contents and depths must be well formatted and structured as shown in the example below in Pascal Case, No lower case.

Example (For all the depths):
This is how it should look like: 
As an example i have put a single skill map showing how each depth must look like. And you must learn and stick to this structure only.
"skill_map": {{
    "title": "Python for Gen AI",
    "progress": 0,
    "depth": 3,
    "contents": {{
      "Python": {{
        "progress": 0,
        "depth": 2,
        "contents": {{
          "numpy": {{
            "progress": 0,
            "depth": 1,
            "contents": {{
              "array operations": {{
                "progress": 0,
                "depth": 0,
                "contents": {{
                  "Vectorized Computations": {{ "progress": 0 }},
                  "Matrix Multiplications": {{ "progress": 0 }},
                  "Statistical Functions": {{ "progress": 0 }},
                  "Indexing and Slicing": {{ "progress": 0 }},
                  "Performance Optimization": {{ "progress": 0 }}
                }}
              }},
              "array manipulation": {{
                "progress": 0,
                "depth": 0,
                "contents": {{
                  "Reshaping Arrays": {{ "progress": 0 }},
                  "Stacking and Splitting": {{ "progress": 0 }},
                  "Merging Arrays": {{ "progress": 0 }},
                  "Filtering and Masking": {{ "progress": 0 }},
                  "Flattening and Transposing": {{ "progress": 0 }}
                }}
              }},
              "broadcasting": {{
                "progress": 0,
                "depth": 0,
                "contents": {{
                  "Broadcasting Rules": {{ "progress": 0 }},
                  "Operations Across Dimensions": {{ "progress": 0 }},
                  "Combining Arrays of Different Shapes": {{ "progress": 0 }},
                  "Performance Impact of Broadcasting": {{ "progress": 0 }},
                  "Common Broadcasting Errors": {{ "progress": 0 }}
                }}
              }}
            }}
          }}
        }}
    }}
}}

"""
        
        input_data = f'<input>{{"skill": "{skill}"}}</input>'
        full_prompt = f"{instructions}\n\nSchema:\n{json.dumps(schema, indent=2)}\n\n{input_data}"
        
        messages = [{"role": "user", "content": [{"text": full_prompt}]}]
        
        #changed model from haiku 4.5 to sonnet 4 cuz haiku was throwing too many tokens error
        response = bedrock_client.converse(
            modelId="apac.anthropic.claude-sonnet-4-20250514-v1:0",
            messages=messages,
            inferenceConfig={"maxTokens": 7000, "temperature": 0.2}
        )
        
        raw_json = response['output']['message']['content'][0]['text']
        parsed = json.loads(raw_json)
        
        chat_title = parsed.get('title', 'Learning Path')
        skill_map_data = parsed.get('skill_map', {})
        skill_map_json = json.dumps(skill_map_data, indent=2)
        
        users_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        f"threads.{thread_id}.skill_map": skill_map_data,
                        f"threads.{thread_id}.chat_title": chat_title,
                        f"threads.{thread_id}.updated_at": time.time(),
                        f"threads.{thread_id}.skill_map_status": "success"
                    }
                },
                upsert=True
            )
        
        print(f"skill_split: Stored skill map in mongo boi for user={user_id}, thread={thread_id}")
        return parsed
    
    except Exception as e:
        print(f"skill_split: Failed - {e}")
        users_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    f"threads.{thread_id}.skill_map_status": "fail"
                }
            },
            upsert=True
        )
        raise

