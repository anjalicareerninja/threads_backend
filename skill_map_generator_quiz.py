import json
import os
import boto3
from pymongo import MongoClient
from dotenv import load_dotenv
import time
load_dotenv()
# from graph_quiz import calculate_progress_upward

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

def skill_split(quiz_data: list, user_goal: str, user_id: str, thread_id: str, skill: str):
    """Generate skill map and store in MongoDB."""
    print(f"skill_split: Starting for user={user_id}, thread={thread_id}, skill={skill}, goal={user_goal}")
    
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
                        "depth": {"type": "integer", "enum": [-1, 0, 1, 2, 3]},
                        "contents": {"type": "object"}
                    },
                    "required": ["title", "progress", "depth", "contents"]
                }
            },
            "required": ["title", "skill_map"]
        }
        
        instructions = """You are an expert strategic and analysis based understanding AI system tasked with generating hierarchical skill maps to structure learning paths. You are to generate a skill map basis the user goal and skill. It must be adaptive to the user's skill level and their goal. Then you see the relevant learnings of the user from the quiz data and intgerate the same into the skill map.

# SKILL MAP RULES:

1. Analyze the skill and the goal provided within <input> tags.
2. Return a JSON response that complies with the provided schema.
3. Do NOT wrap response in markdown code blocks.
4. Output ONLY raw JSON.
---
DEPTH DECISION RULES:
- Depth = -1: The bottom most and most specific level of the skill map. This is the level where the user can learn in depth with very specified topics. 
  Structure: depth -1 root → contents (4-10 as topics to be taught with ONLY {{"progress": 0}})

- Depth = 0: The penultimate level of the skill map. This is the most specific level of the skill map. This is the level where the user can learn the most specific topics. 
  Structure: depth 0 root → depth -1 root (with depth and progress)→ contents(4-10 as topics to be taught with ONLY {{"progress": 0}})

- Depth = 1: Extremely specific topics (4+ words, single action/feature)
  Structure: depth 1 root → depth 0 leaves (with depth and progress) → depth -1 root (with depth and progress)→ contents(4-10 as topics to be taught with ONLY {{"progress": 0}})
  
- Depth = 2: Single technology/domain (1-3 words)
  Structure: depth 2 root → depth 1 topics (with depth and progress) → depth 0 skills (with depth and progress) → depth -1 root (with depth and progress)→ contents (4-10 as topics to be taught with ONLY {{"progress": 0}})
  
- Depth = 3: Multiple domains/techs (contains "and", "for", "with")
  Structure: depth 3 root → depth 2 domains (with depth and progress) → depth 1 topics (with depth and progress) → depth 0 skills (with depth and progress) → depth -1 root (with depth and progress)→ contents (4-10 as topics to be taught with ONLY {{"progress": 0}})
---
HOW TO GENERATE THE SKILL MAP:
1. Understand the depth of the user's skillmap. The user's skill and goal, MUST BE GIVEN focussed while generating a given skill map, think from the user's perspective always. Build a skill map and then incorporate the correct topic_tested (IF RELEVANT ONLY) into it. Likewise for the wrong ones as well. The topic_tested that are wrong, only the relevant ones must be included into the skill map. Just because a topic_tested title is correct doesnt mean it has to be in the skill map to be taught. 
2. Segregrate into well defined contents and subtopics. Map it out clearly and then understand the depth of the skill map. 
3. **CRITICAL**: For each depth=-1 node progress, include ALL and ONLY relevant topics_tested as content headings with progress=X (determine this as you deem fit assesing the user's profile deeply and accurately mind you) if is_correct==1, else progress=0. RELEVANCE IS MUCH MORE IMPORTANT THAN INCLUSIVITY.
4. You must always teach the other user all the neccessary topics for a given skill and goal. Use your extensive knowledge to decide this first and then generate the skill map. You cannot skip any of the vital topics for a given skill map. Do not cram everything into 1 depth heading to make the skill map precise or anything. If a split of that topic is better for the user, then please do split it following the structure rules mentioned above and the example below. 
5. The headings you chose for any of the depths, must be short and concise. Do not make them too long. 
6. The contents inside the depth=-1 node, should not be headings it should be a sentence - an equal muxture of action points - What, How, When, Where, Why (NO BIAS TOWARDS A SINGLE WORD ONLY) etc. So it must be clear enough so that the llm knows what it is to teach, rather an a vague heading without any direction, on what is to be taught to make sure the llm knows whhat should be taught. 
---
CRITICAL STRUCTURE RULES:
- EVERY node MUST have "progress" and "depth" keys EXCEPT the final leaf subskills
- You must reason with every single content heading and depth headings you will be including into the skill map. Because it has to make complete sense for the skill the user wants to learn. So the end goal is, every single aspect of your skill map generated should make sense to the user. You are not allowed to include uneccessary topics here mind you. 
- Parent depth MUST be EXACTLY 1 greater than child depth (depth 3 → depth 2 → depth 1 → depth 0 → depth -1)
- Final leaf subskills (inside depth=-1 contents) ONLY have {{"progress": 0}} - NO depth key
- Depth=-1 nodes MUST have 3-6 specific subskills (STRICT BOUNDARY RULE) in their contents dictionary
- All names must be unique and relevant
- NEVER skip intermediate depth levels - if root is depth=2, you MUST have depth=1 nodes, then depth=0 nodes
- The skill map and all its contents and depths must be well formatted and structured as shown in the example below in Pascal Case, No lower case.
- The title must only be max 2-3 WORD ONLY. 

# HOW AND WHEN TO INTEGRATE QUIZ DATA: 

**CRITICAL: QUIZ DATA IS ONLY FOR SETTING PROGRESS VALUES, NOT FOR ADDING TOPICS**

STEP 1: Analyse all the data given as quiz_data. There are certain rules you must follow:
a. For every question, you must understand the user's learning capability, answering capability and understanding of that certain topic. Basis that is what you're to include that topic_tested into the skill map, **if and ONLY if it's CORE to the skill and goal**.
b. Not only must you pay attention to the topic_tested, but also the quiz questions and user's answers as well. 

STEP 2: After building a skill map, for the skill and the goal the user wants, incorporate the relevant quiz data ONLY into the skill map. The skill map takes precedence over the quiz data. You give skillmap FIRST AND ONLY, if relevant should you include the quiz data into the skill map. 
  **CRITICAL:** SKILL MAP >>>> QUIZ DATA

**EXCLUSION RULES - DO NOT ADD THESE TOPICS UNLESS EXPLICITLY IN SKILL/GOAL:**

For "Java OOP" or "OOP basics" or "OOP for interviews":
- ❌ DO NOT include: SOLID Principles (SRP, OCP, LSP, ISP, DIP)
- ❌ DO NOT include: Design Patterns (Strategy, Factory, Singleton, etc.)
- ❌ DO NOT include: Best Practices, Code Quality, Maintainability
- ❌ DO NOT include: Advanced Design topics
- ✅ ONLY include: Classes, Objects, Encapsulation, Inheritance, Polymorphism, Abstraction (the 4 pillars)

For "Advanced Java Design" or "Software Architecture":
- ✅ THEN include SOLID, Design Patterns, etc.

**IF QUIZ HAS SRP/SOLID QUESTIONS BUT SKILL IS "JAVA OOP":**
- Find matching core OOP concept (e.g., SRP → Encapsulation)
- Set progress in that core concept's leaf contents
- **DO NOT create new "SOLID Principles" or "Design Principles" depth nodes**

STEP 3: For EACH relevant topic_tested, find the appropriate depth=-1 node where it belongs amongst the other depth=-1 nodes, basis which user got the answer correct and incorportaing his relevant score update for that known topic inside depth -1. 

STEP 4: Add that topic_tested as a content heading (**ONLY IF DEEMED CORE AND ESSENTIAL TO THE SKILL AND GOAL OF THE USER**) under that depth=0 node if relevant ONLY. Mind you that the skill map is oblivious to the topics_tested that the user has gotten correct. It is just an overlay on the skill map not an integration into it or anything. The goal is to show the pre existing relevant knowledge of the user thats it for that skill. NOTHING MORE.

STEP 5: **CRITICAL - PRECISE PROGRESS MAPPING:**

When is_correct==1, you MUST analyze the question content deeply and set progress ONLY for the specific leaf contents that were actually tested.

**DO NOT set progress for all leaf contents under a depth=-1 node just because the topic_tested matches!**

**Example of CORRECT behavior:**
- Quiz: "Which OOP concept focuses on protecting an object's internal state?" → Answer: "Encapsulation" ✅
- topic_tested: "Encapsulation Fundamentals"
- Question tests: ONLY the basic definition/concept of encapsulation
- **Set progress ONLY for:** "Data Hiding Principles": 40 (since question is about protecting state)
- **DO NOT set progress for:** Access Modifiers, Getter/Setter Methods, etc. (not tested in this question)

**Example of WRONG behavior:**
- ❌ Setting progress for ALL leaf contents (Data Hiding: 60, Access Modifiers: 50, Getters: 40, etc.)
- This is WRONG because the question only tested the basic concept, not the specific implementation details

**How to determine which leaf contents to update:**
1. Read the question text carefully
2. Identify EXACTLY what knowledge was tested
3. Match that to 1-2 specific leaf contents (maximum)
4. Set progress ONLY for those matched contents
5. Leave all other leaf contents at progress=0

**More examples:**
- Question: "What is private keyword used for?" → Set progress for "Access Modifiers" only
- Question: "Why use getter methods?" → Set progress for "Getter Setter Methods" only  
- Question: "What is encapsulation?" → Set progress for "Data Hiding Principles" or "Information Hiding" only

STEP 6: Set progress=0 for topics where is_correct==0 (user got it wrong)



EXAMPLE:
If quiz_data has:
- {"topic_tested": "Matrix Multiplications", "is_correct": 1}
- {"topic_tested": "Performance Optimization", "is_correct": 0}
- {"topic_tested": "Vectorized Computations", "is_correct": 0}

Then your skill map MUST include, ONLY THE RELEVANT correct topic_tested titles for the user skill:
```
"NumPy Operations": {
  "progress": Y,
  "depth": 0,
  "contents": {

    "Matrix Multiplications": {
      "progress": X,
      "depth": -1,
      "contents": {
        "What does dimensionality mean in NumPy arrays and how does it affect matrix multiplication?": { "progress": a },
        "How is dot product different from matrix multiplication in NumPy?": { "progress": b },
        "What is the @ operator and how does it differ from np.dot()?": { "progress": c },
        "When should np.dot() be used vs np.matmul(), especially with higher-dimensional arrays?": { "progress": d },
        "How do transpose and axis swapping change matrix multiplication behavior?": { "progress": e },
        "How does NumPy perform batch matrix multiplication for 3D and higher tensors?": { "progress": f },
        "What are identity matrices, inverses, and rank, and why do they matter in linear algebra operations?": { "progress": g },
        "How are common linear algebra operations implemented using np.linalg?": { "progress": h },
        "What numerical stability issues arise in matrix operations and how does NumPy handle them?": { "progress": i },
        "What are the most common shape and broadcasting errors in matrix multiplication and how can they be debugged?": { "progress": j }
      }
    },

    "Performance Optimization": {
      "progress": 0,
      "depth": -1,
      "contents": {
        "Why are Python loops slower than NumPy operations under the hood?": { "progress": 0 },
        "What does vectorization mean in NumPy and why does it improve performance?": { "progress": 0 },
        "How do memory layout and strides affect NumPy performance?": { "progress": 0 },
        "When should in-place operations be used and what are their trade-offs?": { "progress": 0 },
        "How do temporary arrays get created unintentionally and how can they be avoided?": { "progress": 0 },
        "How expensive is broadcasting and when does it become a performance bottleneck?": { "progress": 0 },
        "How do different dtypes (float32 vs float64) impact speed and memory usage?": { "progress": 0 },
        "How does CPU cache efficiency influence NumPy computation speed?": { "progress": 0 },
        "How can NumPy code be profiled to identify performance bottlenecks?": { "progress": 0 },
        "How should time and memory complexity be reasoned about in NumPy-based pipelines?": { "progress": 0 }
      }
    },

    "Vectorized Computations": {
      "progress": 0,
      "depth": -1,
      "contents": {
        "How do element-wise operations work in NumPy compared to scalar operations?": { "progress": 0 },
        "What are the broadcasting rules and how does NumPy apply them automatically?": { "progress": 0 },
        "How does boolean masking work and why is it preferred over conditional loops?": { "progress": 0 },
        "How do reduction operations behave across different axes in multi-dimensional arrays?": { "progress": 0 },
        "How can conditional logic be vectorized using NumPy operations?": { "progress": 0 },
        "How can multiple vectorized operations be chained without creating unnecessary arrays?": { "progress": 0 },
        "Why should explicit for-loops be avoided in NumPy and when are they unavoidable?": { "progress": 0 },
        "What numerical stability issues arise in vectorized computations and how can they be mitigated?": { "progress": 0 },
        "How is vectorization used in real-world feature engineering pipelines?": { "progress": 0 },
        "How can NumPy code be written to closely resemble mathematical notation for readability?": { "progress": 0 }
      }
    }

  }
  ...
}

```
---


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
                  "Vectorized Computations": {
                    "progress": 0,
                    "depth": -1,
                    "contents": {
          "What element-wise operations are and how NumPy applies them across arrays": { "progress": 0 },
          "Why broadcasting works and when NumPy applies broadcasting rules": { "progress": 0 },
          "How boolean masking enables conditional selection without loops": { "progress": 0 },
          "What reduction operations do and when axis selection matters": { "progress": 0 },
          "Why replacing Python loops with vectorized logic improves performance": { "progress": 0 },
          "How to structure chained vectorized operations for readable code": { "progress": 0 }
        }
                  },
                  "Matrix Multiplications": {
                    "progress": 16,
                    "depth": -1,
                    "contents": {
          "What array dimensionality means and how it governs matrix multiplication": { "progress": 10 },
          "How dot product behavior differs from matrix multiplication in NumPy": { "progress": 45 },
          "When to use the @ operator for explicit matrix multiplication": { "progress": 0 },
          "Why np.dot and np.matmul behave differently for higher dimensions": { "progress": 0 },
          "How transpose and axis alignment affect multiplication results": { "progress": 0 },
          "When batch matrix multiplication is required and how NumPy performs it": { "progress": 60 },
          "How common linear algebra operations are accessed through np.linalg": { "progress": 0 }
        }
                  },
                  "Statistical Functions": "Statistical Functions": {
                    "progress": 0,
                    "depth": -1,
                    "contents": {
          "What mean, median, and variance represent in numerical datasets": { "progress": 0 },
          "How standard deviation is computed and interpreted in NumPy": { "progress": 0 },
          "When statistical computations should be applied along specific axes": { "progress": 0 },
          "How NumPy identifies minimum and maximum values with indices": { "progress": 0 },
          "Why numerical stability issues arise in statistical calculations": { "progress": 0 }
        }
                  },
                  "Indexing and Slicing": "Indexing and Slicing": {
                    "progress": 0,
                    "depth": -1,
                    "contents": {
          "How basic indexing retrieves individual elements from arrays": { "progress": 0 },
          "What slicing semantics return when extracting subarrays": { "progress": 0 },
          "When boolean indexing should be used over conditional loops": { "progress": 0 },
          "How fancy indexing differs from slicing in behavior and cost": { "progress": 0 },
          "Why some indexing operations return views instead of copies": { "progress": 0 },
          "What common logical errors occur in advanced indexing scenarios": { "progress": 0 }
        }
                  },
                  "Performance Optimization": "Performance Optimization": {
                    "progress": 0,
                    "depth": -1,
                    "contents": {
          "Why NumPy operations outperform equivalent Python loops": { "progress": 0 },
          "When in-place operations should be used to reduce memory usage": { "progress": 0 },
          "How memory layout and strides influence execution performance": { "progress": 0 },
          "Why unnecessary temporary arrays degrade performance": { "progress": 0 },
          "How choosing appropriate data types improves efficiency": { "progress": 0 },
          "When profiling NumPy code is necessary to find bottlenecks": { "progress": 0 }
        }
                  }

                }}
              }},
              "array manipulation": {{
                "progress": 0,
                "depth": 0,
                "contents": {{
                  "Reshaping Arrays": {{ "progress": 0, "depth": -1, "contents": {...}}},
                  "Stacking and Splitting": {{ "progress": 0, "depth": -1, "contents": {...}}},
                  "Merging Arrays": {{ "progress": 0, "depth": -1, "contents": {...}}},
                  "Filtering and Masking": {{ "progress": 0, "depth": -1, "contents": {...}}},
                  "Flattening and Transposing": {{ "progress": 0, "depth": -1, "contents": {...}}}
                }}
              }},
              "broadcasting": {{
                "progress": 0,
                "depth": 0,
                "contents": {{
                  "Broadcasting Rules": {{ "progress": 0, "depth": -1, "contents": {...}}},
                  "Operations Across Dimensions": {{ "progress": 0, "depth": -1, "contents": {...}}},
                  "Combining Arrays of Different Shapes": {{ "progress": 0, "depth": -1, "contents": {...}}},
                  "Performance Impact of Broadcasting": {{ "progress": 0, "depth": -1, "contents": {...}}},
                  "Common Broadcasting Errors": {{ "progress": 0, "depth": -1, "contents": {...}}}
                }}
              }}
            }}
          }}
        }}
    }}
}}

Assuming that the topics of Matrix Multiplication, Filtering and Masking and Broadcasting Rules are some of the topics_tested that had its is_correct==1 such that the skill the user wants to learn is about numpy broadcasting majorly i.e. the user got that topic right with that particular progress value = (A value that you compute as shown in the example for Matrix Multiplication topic, such that you analyse what the user knows and doesnt basis the question and its option and the correct answer chosen by the user). Similary, Combining Arrays of Differnt Shapes are wrongly answered by the user. But mind you that off the quiz data despite having a bunch of other topics_tested values, but only Broadcasting Rules and Combining Arrays of Different Shapes are relevant to the skill the user wants to learn and hence ONLY those are included in the skill map. Yet BOTH types are embedded into the skill map and it reflects the learning and understanding of the user as well. Learn from this example and thats exactly how you are to generate a skill map ONLY.

NOW FOLLOW THE ABOVE RULES TO THE 'T'.

"""
        
        skill = f'<input>{{"skill": "{skill}"}}</input>'
        quiz_data = f'<input>{{"quiz_data": "{quiz_data}"}}</input>'
        user_goal_a = f'<input>{{"user_goal": "{user_goal}"}}</input>'
        full_prompt = f"{instructions}\n\nSchema:\n{json.dumps(schema, indent=2)}\n\n{skill}\n{quiz_data}\n{user_goal_a}"
        # print(f'full_prompt from skill map bg file: {full_prompt}')
        
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
        # print(f'skill map data 175 {skill_map_data}')
        # print(f'skill map json 175 {skill_map_json}')
        def calculate_progress_upward(node):
                        """Recursively calculate progress as average of children"""
                        if not isinstance(node, dict):
                            return 0
                        
                        # If node has contents, calculate average of children
                        if 'contents' in node and isinstance(node['contents'], dict) and node['contents']:
                            child_progresses = []
                            for child_key, child_value in node['contents'].items():
                                if isinstance(child_value, dict):
                                    # Check if it's a simple sub-skill (no depth field) or a nested node
                                    if 'depth' not in child_value:
                                        # Simple sub-skill: just {"progress": X}
                                        child_progresses.append(child_value.get('progress', 0))
                                    else:
                                        # Nested node with depth - recursively calculate first
                                        child_progress = calculate_progress_upward(child_value)
                                        child_progresses.append(child_progress)
                            
                            # Average of children (integer division)
                            if child_progresses:
                                node['progress'] = sum(child_progresses) // len(child_progresses)
                                return node['progress']
                        
                        # If no contents (leaf node), return its own progress
                        return node.get('progress', 0)
        calculate_progress_upward(skill_map_data)
        
        c=user_goal_a.find(user_goal)
        # print(f'iron throne: {c}')
        d=user_goal_a[c:c+len(user_goal)]
        # print(f'kings landing: {d}')

        users_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        f"threads.{thread_id}.skill_map": skill_map_data,
                        f"threads.{thread_id}.chat_title": chat_title,
                        f"threads.{thread_id}.updated_at": time.time(),
                        f"threads.{thread_id}.skill_map_status": "success",
                        f"threads.{thread_id}.user_goal": d
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

