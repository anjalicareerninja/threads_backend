from ast import List
from typing import Any
from langchain.agents.middleware import before_agent, after_agent
from langgraph.runtime import Runtime
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.config import get_config
from cerebras.cloud.sdk import Cerebras


import os
import json
import google.genai as genai
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.output_parsers import PydanticToolsParser
from typing_extensions import TypedDict
from typing import Annotated
import operator
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langgraph.types import interrupt
from pymongo import MongoClient
import uuid
import time
import boto3
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_cerebras import ChatCerebras
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, Union, Any, Optional

from langgraph.checkpoint.mongodb import MongoDBSaver
from apscheduler.schedulers.background import BackgroundScheduler
# from skill_map_generator import skill_split as skill_split_bg
from skill_map_generator_quiz import skill_split as skill_split_bg



load_dotenv()

scheduler = BackgroundScheduler()
scheduler.start()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "guardrails"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0.3,
#     stream=True,
#     streaming_chunk_size=1000,
#     _should_stream=True,
# )


bedrock_client = boto3.client(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    service_name="bedrock-runtime",
    region_name="ap-south-1"
)

chats = [
    {
        "role": "user",
        "content": [{"text": "Your message here"}]
    }
]

# LangChain-compatible Bedrock model for use with create_agent
bedrock_llm = ChatBedrockConverse(
    model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
    client=bedrock_client,
    temperature=0.2,
    max_tokens=6000, 
    additional_model_request_fields={
        "anthropic_beta": ["context-1m-2025-08-07"]  # 1m context thing
    },
    supports_tool_choice_values=("auto", "any", "tool") 
)

bedrock_llm_sonnet = ChatBedrockConverse(
    model_id="apac.anthropic.claude-sonnet-4-20250514-v1:0",
    client=bedrock_client,
    temperature=0.2,
    max_tokens=2000, 
    additional_model_request_fields={
        "anthropic_beta": ["context-1m-2025-08-07"]  # 1m context thing
    },
    supports_tool_choice_values=("auto", "any", "tool") 
)

bedrock_llm_sonnet_new = ChatBedrockConverse(
    model_id="apac.anthropic.claude-sonnet-4-20250514-v1:0",
    client=bedrock_client,
    temperature=0.2,
    max_tokens=6000, 
    additional_model_request_fields={
        "anthropic_beta": ["context-1m-2025-08-07"]  # 1m context thing
    },
    supports_tool_choice_values=("auto", "any", "tool") 
)

cerebras_llm = ChatCerebras(
    model="llama-3.3-70b",
    api_key=os.getenv("CEREBRAS_API_KEY"),
    temperature=0.1,
    max_tokens=10000,
    max_retries=2,
)

cerebras_client = Cerebras(
    api_key=os.getenv("CEREBRAS_API_KEY")
)

# mongodb part
try:
    mongo_url = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db_name = "chatbot"
    db = client[db_name]
    chats_collection = db['chats']
    users_collection = db['users']
    memory = MongoDBSaver(client=client, db_name=db_name, collection_name="checkpoints")
    print("MongoDB persistence initialized.")
except Exception as e:
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    chats_collection = None
    users_collection = None
    print(f"Falling back to in-memory persistence: {e}")



#storing chats in mongodb
def store_chats(
    thread_id: str, messages: list[BaseMessage], metadata: dict = None
):
    """Store/update conversation messages in the chats collection"""
    if chats_collection is None:
        print("MongoDB not available, skipping conversation storage")
        return

    try:
        # Serialize BaseMessage objects to dictionaries
        serializable_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            msg_dict = {"role": role, "content": msg.content}
            
            # CRITICAL FIX: Store follow_up_questions and quiz data if present in additional_kwargs
            # NOTE: block flag is NOT stored in DB - handled only in response/SSE stream
            if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
                follow_ups = msg.additional_kwargs.get('follow_up_questions')
                if follow_ups and isinstance(follow_ups, list):
                    msg_dict["follow_up_questions"] = follow_ups
                
                # Store quiz_available if present
                quiz_available = msg.additional_kwargs.get('quiz_available')
                if quiz_available and isinstance(quiz_available, dict):
                    msg_dict["quiz_available"] = quiz_available
                
                # Store sidebar_quizzes if present
                sidebar_quizzes = msg.additional_kwargs.get('sidebar_quizzes')
                if sidebar_quizzes and isinstance(sidebar_quizzes, list):
                    msg_dict["sidebar_quizzes"] = sidebar_quizzes
                
                # Store quiz data if present
                quiz = msg.additional_kwargs.get('quiz')
                if quiz and isinstance(quiz, dict):
                    msg_dict["quiz"] = quiz
            
            serializable_messages.append(msg_dict)

        chats_collection.update_one(
            {"thread_id": thread_id},
            {
                "$push": {"messages": {"$each": serializable_messages}},
                "$setOnInsert": {
                    "thread_id": thread_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            },
            upsert=True,
        )
        print(f"Conversation messages stored for thread {thread_id}")

    except Exception as e:
        print(f"Failed to store conversation message: {e}")

# storing skill map in users collection per thread
def store_skill_map(
    thread_id: str, user_id: str, skill_map: str, chat_title: str
):
    """Store/update skill map in users collection for a specific thread"""
    if users_collection is None:
        print("MongoDB not available, skipping skill map storage")
        return
    
    try:
        # Parse skill_map to store as dict
        skill_map_dict = json.loads(skill_map) if isinstance(skill_map, str) else skill_map
        
        # Update or insert user document with thread-specific skill map
        users_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    f"threads.{thread_id}.skill_map": skill_map_dict,
                    f"threads.{thread_id}.chat_title": chat_title,
                    f"threads.{thread_id}.updated_at": time.time()
                }
            },
            upsert=True
        )
        print(f"Skill map stored for user {user_id}, thread {thread_id}")
        
    except Exception as e:
        print(f"Failed to store skill map: {e}")

def update_depth_zero_progress(thread_id: str, user_id: str, topic: str, progress: int):
    """
    Update the progress value of a specific depth=0 node in the skill map.
    
    Args:
        thread_id: Thread ID
        user_id: User ID
        topic: Name of the depth=0 node to update
        progress: New progress value (typically 95 when quiz is passed)
    """
    if users_collection is None:
        print("MongoDB not available, skipping skill map progress update")
        return
    
    try:
        # Get current skill map from MongoDB
        user_doc = users_collection.find_one({"user_id": user_id})
        if not user_doc:
            print(f"User {user_id} not found")
            return
        
        threads = user_doc.get("threads", {})
        thread_data = threads.get(thread_id, {})
        skill_map_dict = thread_data.get("skill_map")
        
        if not skill_map_dict:
            print(f"❌ Skill map not found for thread_id='{thread_id}'")
            print(f"   Available threads: {list(threads.keys())}")
            if thread_id in threads:
                print(f"   Thread data keys: {list(thread_data.keys())}")
            return
        
        print(f"🔍 Searching for depth=0 node '{topic}' in skill map...")
        print(f"   Skill map structure: {list(skill_map_dict.keys()) if isinstance(skill_map_dict, dict) else 'not a dict'}")
        
        # Recursively find and update the depth=0 node with matching topic name
        # Handle case-insensitive matching and trimmed strings
        topic_normalized = topic.strip().lower()
        
        def find_and_update_node(node, target_topic, target_topic_normalized, new_progress, path=""):
            """Recursively search for depth=0 node with matching name and update progress"""
            if isinstance(node, dict):
                # If this node has contents, search through them
                if "contents" in node:
                    for key, value in node["contents"].items():
                        current_path = f"{path}.{key}" if path else key
                        key_normalized = key.strip().lower()
                        
                        # Check if this key matches the topic name (case-insensitive) and is depth=0
                        if key_normalized == target_topic_normalized and isinstance(value, dict):
                            if value.get("depth") == 0:
                                # Found the target depth=0 node
                                value["progress"] = new_progress
                                print(f"✅ Updated progress for depth=0 node '{key}' (matched '{target_topic}') to {new_progress}")
                                return True
                        # Recursively search in nested contents
                        if isinstance(value, dict):
                            if find_and_update_node(value, target_topic, target_topic_normalized, new_progress, current_path):
                                return True
            return False
        
        # Start searching from skill_map root
        skill_map_root = skill_map_dict.get("skill_map", skill_map_dict)
        updated = False
        
        # Search in contents - check both exact match and normalized match
        if "contents" in skill_map_root:
            for key, value in skill_map_root["contents"].items():
                key_normalized = key.strip().lower()
                if (key == topic or key_normalized == topic_normalized) and isinstance(value, dict) and value.get("depth") == 0:
                    value["progress"] = progress
                    updated = True
                    print(f"✅ Updated progress for depth=0 node '{key}' (matched '{topic}') to {progress}")
                    break
                elif isinstance(value, dict):
                    if find_and_update_node(value, topic, topic_normalized, progress):
                        updated = True
                        break
        
        if updated:
            # CRITICAL: Recalculate progress upward (average from bottom to top)
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
            
            # Recalculate all parent nodes' progress upward
            calculate_progress_upward(skill_map_root)
            print(f"✅ Progress recalculated upward after updating '{topic}' to {progress}")
            
            # Save updated skill map back to MongoDB
            users_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        f"threads.{thread_id}.skill_map": skill_map_dict,
                        f"threads.{thread_id}.updated_at": time.time()
                    }
                }
            )
            print(f"✅ Skill map updated in MongoDB for topic: {topic} (with upward recalculation)")
        else:
            print(f"⚠️ Could not find depth=0 node '{topic}' in skill map to update progress")
    
    except Exception as e:
        print(f"Failed to update depth=0 progress: {e}")

# storing quiz in users collection per thread
def store_quiz(
    thread_id: str, user_id: str, quiz_id: str, topic: str, quiz_data: dict, should_append: bool = False
):
    """Store/update quiz in users collection for a specific thread and topic
    New architecture: threads.{thread_id}.quizzes.{topic} = [quiz1, quiz2, ...]
    Each quiz object contains quiz_id, topic, questions, score, status
    """
    if users_collection is None:
        print("MongoDB not available, skipping quiz storage")
        return
    
    try:
        # Add score and user_answer fields to each question
        # CRITICAL: Ensure correct_answer is always stored as integer
        questions_with_tracking = []
        for q in quiz_data.get("questions", []):
            question_with_tracking = q.copy()
            question_with_tracking["user_answer"] = None  # Initialize as None
            # Ensure correct_answer is stored as integer (handle string/number mismatches)
            if "correct_answer" in question_with_tracking:
                try:
                    question_with_tracking["correct_answer"] = int(question_with_tracking["correct_answer"])
                except (ValueError, TypeError):
                    print(f"⚠️ Warning: Could not convert correct_answer to int for question: {q.get('question', 'unknown')}")
            questions_with_tracking.append(question_with_tracking)
        
        # Create quiz structure with quiz_id included in the data
        quiz_storage_data = {
            "quiz_id": quiz_id,
            "topic": quiz_data.get("topic", topic),
            "questions": questions_with_tracking,
            "score": 0,  # Initialize score as 0
            "status": "unattempted"  # unattempted, passed, failed
        }
        
        # New architecture: threads.{thread_id}.quizzes.{topic} = [quiz1, quiz2, ...]
        if should_append:
            # Append new quiz to array (for failed quiz retry)
            users_collection.update_one(
                {"user_id": user_id},
                {
                    "$push": {
                        f"threads.{thread_id}.quizzes.{topic}": quiz_storage_data
                    },
                    "$set": {
                        f"threads.{thread_id}.updated_at": time.time()
                    }
                },
                upsert=True
            )
            print(f"✅ Quiz appended to array for user {user_id}, thread {thread_id}, topic: {topic}, quiz_id: {quiz_id}")
        else:
            # Replace unattempted quiz or create new array
            # First, check if topic exists and has unattempted quiz
            user_doc = users_collection.find_one({"user_id": user_id})
            topic_quizzes = []
            if user_doc:
                threads = user_doc.get("threads", {})
                thread_data = threads.get(thread_id, {})
                quizzes = thread_data.get("quizzes", {})
                topic_quizzes = quizzes.get(topic, [])
            
            # If array exists and last quiz is unattempted, replace it
            if topic_quizzes and len(topic_quizzes) > 0:
                last_quiz = topic_quizzes[-1]
                if last_quiz.get("status") == "unattempted":
                    # Replace the last unattempted quiz
                    users_collection.update_one(
                        {"user_id": user_id},
                        {
                            "$set": {
                                f"threads.{thread_id}.quizzes.{topic}.{len(topic_quizzes) - 1}": quiz_storage_data,
                                f"threads.{thread_id}.updated_at": time.time()
                            }
                        }
                    )
                    print(f"✅ Replaced unattempted quiz for user {user_id}, thread {thread_id}, topic: {topic}, quiz_id: {quiz_id}")
                else:
                    # Append if last quiz is not unattempted
                    users_collection.update_one(
                        {"user_id": user_id},
                        {
                            "$push": {
                                f"threads.{thread_id}.quizzes.{topic}": quiz_storage_data
                            },
                            "$set": {
                                f"threads.{thread_id}.updated_at": time.time()
                            }
                        }
                    )
                    print(f"✅ Quiz appended (last quiz was not unattempted) for user {user_id}, thread {thread_id}, topic: {topic}, quiz_id: {quiz_id}")
            else:
                # Create new array with first quiz
                users_collection.update_one(
                    {"user_id": user_id},
                    {
                        "$set": {
                            f"threads.{thread_id}.quizzes.{topic}": [quiz_storage_data],
                            f"threads.{thread_id}.updated_at": time.time()
                        }
                    },
                    upsert=True
                )
                print(f"✅ Created new quiz array for user {user_id}, thread {thread_id}, topic: {topic}, quiz_id: {quiz_id}")
        
    except Exception as e:
        print(f"Failed to store quiz: {e}")

# update quiz answers and calculate score
def update_quiz_answers(
    thread_id: str, user_id: str, quiz_id: str, topic: str, answers: list[dict]
):
    """
    Update user answers for a quiz and calculate score.
    
    Args:
        thread_id: Thread ID
        user_id: User ID
        topic: Quiz topic name
        answers: List of dicts with question_index and user_answer
                Example: [{"question_index": 0, "user_answer": 2}, ...]
    
    Returns:
        dict with updated quiz data including score
    """
    if users_collection is None:
        print("MongoDB not available, skipping quiz update")
        return None
    
    try:
        # Get current quiz
        user_doc = users_collection.find_one({"user_id": user_id})
        if not user_doc:
            print(f"User {user_id} not found")
            return None
        
        # New architecture: threads.{thread_id}.quizzes.{topic} = [quiz1, quiz2, ...]
        threads = user_doc.get("threads", {}) if isinstance(user_doc, dict) else {}
        thread_data = threads.get(thread_id, {}) if isinstance(threads, dict) else {}
        quizzes = thread_data.get("quizzes", {}) if isinstance(thread_data, dict) else {}
        
        # Ensure quizzes is a dict before accessing
        if not isinstance(quizzes, dict):
            print(f"❌ Quizzes is not a dict: {type(quizzes)}")
            return None
        
        topic_quizzes_raw = quizzes.get(topic, []) if isinstance(quizzes, dict) else []
        
        # Handle migration: if old format (dict), convert to array format
        if isinstance(topic_quizzes_raw, dict):
            print(f"⚠️ Old format detected for topic '{topic}', converting to array format")
            topic_quizzes = []
            for qid, qdata in topic_quizzes_raw.items():
                if isinstance(qdata, dict):
                    qdata_copy = qdata.copy()
                    qdata_copy["quiz_id"] = qid  # Ensure quiz_id is in the data
                    topic_quizzes.append(qdata_copy)
            # Update MongoDB with new format
            users_collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        f"threads.{thread_id}.quizzes.{topic}": topic_quizzes
                    }
                }
            )
            print(f"✅ Converted old format to array format for topic '{topic}'")
        elif isinstance(topic_quizzes_raw, list):
            topic_quizzes = topic_quizzes_raw
        else:
            topic_quizzes = []
        
        if not topic_quizzes or len(topic_quizzes) == 0:
            print(f"❌ Quiz not found: topic='{topic}', quiz_id='{quiz_id}' for thread_id='{thread_id}'")
            print(f"   Available threads: {list(threads.keys())}")
            if thread_id in threads:
                print(f"   Available quiz topics in thread: {list(quizzes.keys())}")
            return None
        
        # Find quiz by quiz_id in the array using list indexing
        quiz_data = None
        quiz_index = None
        for idx in range(len(topic_quizzes)):
            quiz = topic_quizzes[idx]
            # Ensure quiz is a dict before accessing it
            if not isinstance(quiz, dict):
                continue
            # Check quiz_id using direct dict access, not .get()
            if "quiz_id" in quiz and quiz["quiz_id"] == quiz_id:
                quiz_data = quiz.copy()
                quiz_index = idx
                break
        
        if not quiz_data or quiz_index is None:
            print(f"❌ Quiz not found: topic='{topic}', quiz_id='{quiz_id}' for thread_id='{thread_id}'")
            if topic in quizzes:
                available_quiz_ids = []
                for q in topic_quizzes:
                    if isinstance(q, dict) and "quiz_id" in q:
                        available_quiz_ids.append(q["quiz_id"])
                print(f"   Available quiz_ids for topic '{topic}': {available_quiz_ids}")
            return None
        
        print(f"✅ Found quiz at index {quiz_index} for topic '{topic}', quiz_id: {quiz_id}")
        
        # Update user answers - access questions directly from dict, not using .get()
        if "questions" not in quiz_data:
            print(f"❌ Questions key missing in quiz_data")
            return None
        
        questions = quiz_data["questions"]
        if not isinstance(questions, list):
            print(f"❌ Questions is not a list: {type(questions)}")
            return None
        
        # Make a copy of questions list to modify
        questions = [q.copy() if isinstance(q, dict) else q for q in questions]
        
        for answer_data in answers:
            if not isinstance(answer_data, dict):
                continue
            # Use direct dict access, not .get()
            q_index = answer_data["question_index"] if "question_index" in answer_data else None
            user_answer = answer_data["user_answer"] if "user_answer" in answer_data else None
            
            if q_index is not None and isinstance(q_index, int) and 0 <= q_index < len(questions):
                question = questions[q_index]
                if isinstance(question, dict):
                    question["user_answer"] = user_answer
                else:
                    print(f"⚠️ Question at index {q_index} is not a dict: {type(question)}")
        
        # Calculate score - STRICT TYPE-SAFE COMPARISON
        correct_count = 0
        for q in questions:
            if not isinstance(q, dict):
                continue
            # Use direct dict access, not .get()
            user_answer = q["user_answer"] if "user_answer" in q else None
            correct_answer = q["correct_answer"] if "correct_answer" in q else None
            
            # Convert both to integers for strict comparison (handle string/number mismatches)
            if user_answer is not None and correct_answer is not None:
                try:
                    user_answer_int = int(user_answer)
                    correct_answer_int = int(correct_answer)
                    if user_answer_int == correct_answer_int:
                        correct_count += 1
                        question_idx = q["question_index"] if isinstance(q, dict) and "question_index" in q else "unknown"
                        print(f"✅ Question {question_idx}: Correct (user={user_answer_int}, correct={correct_answer_int})")
                    else:
                        question_idx = q["question_index"] if isinstance(q, dict) and "question_index" in q else "unknown"
                        print(f"❌ Question {question_idx}: Incorrect (user={user_answer_int}, correct={correct_answer_int})")
                except (ValueError, TypeError) as e:
                    question_idx = q["question_index"] if isinstance(q, dict) and "question_index" in q else "unknown"
                    print(f"⚠️ Error comparing answers for question {question_idx}: {e} (user_answer={user_answer}, correct_answer={correct_answer})")
        
        score = correct_count
        total_questions = len(questions)
        
        # Determine status: passed if score > 7, otherwise failed
        status = "passed" if score > 7 else "failed"
        
        # CRITICAL: Count wrong answers per content_title for adaptive quiz generation
        # This helps focus retry quizzes on areas where user struggled
        content_title_wrong_counts = {}
        for q in questions:
            if not isinstance(q, dict):
                continue
            # Check if answer is wrong
            user_answer = q.get("user_answer")
            correct_answer = q.get("correct_answer")
            is_wrong = False
            
            if user_answer is not None and correct_answer is not None:
                try:
                    user_answer_int = int(user_answer)
                    correct_answer_int = int(correct_answer)
                    if user_answer_int != correct_answer_int:
                        is_wrong = True
                except (ValueError, TypeError):
                    pass
            
            # If wrong, count it for the content_title
            if is_wrong:
                # Get content_title from the question itself (not from options)
                content_title = q.get("content_title")
                if content_title:
                    content_title_wrong_counts[content_title] = content_title_wrong_counts.get(content_title, 0) + 1
        
        print(f"📊 Wrong answer counts per content_title: {content_title_wrong_counts}")
        
        # Update quiz in MongoDB - New architecture: threads.{thread_id}.quizzes.{topic}[index]
        # Store content_title_wrong_counts in quiz data for adaptive retry quizzes
        update_fields = {
            f"threads.{thread_id}.quizzes.{topic}.{quiz_index}.questions": questions,
            f"threads.{thread_id}.quizzes.{topic}.{quiz_index}.score": score,
            f"threads.{thread_id}.quizzes.{topic}.{quiz_index}.status": status,
            f"threads.{thread_id}.updated_at": time.time()
        }
        
        # Only store wrong counts if there are any wrong answers
        if content_title_wrong_counts:
            update_fields[f"threads.{thread_id}.quizzes.{topic}.{quiz_index}.content_title_wrong_counts"] = content_title_wrong_counts
        
        users_collection.update_one(
            {"user_id": user_id},
            {"$set": update_fields}
        )
        
        print(f"✅ Quiz answers updated - Score: {score}/{total_questions}, Status: {status} for topic: {topic}")
        
        # NOTE: Progress is calculated upward from children, so we don't manually set depth=0 progress
        # The progress_tracker tool will handle progress updates based on what's actually taught
        
        # Return updated quiz data
        return {
            "topic": topic,
            "questions": questions,
            "score": score,
            "total_questions": total_questions,
            "status": status
        }
        
    except Exception as e:
        print(f"Failed to update quiz answers: {e}")
        return None



# agent state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    skill_map: str
    chat_title: str
    # title: str
    thread_id: str  # For MongoDB updates
    user_id: str  # For MongoDB updates
    current_topic: str 
    block: bool # to see if its a block or not
    quiz_data: list 
    user_goal: str
    skill: str

# Structured output models for skill map
class LeafNode(BaseModel):
    """Leaf node (skill) with progress and depth. 
    For depth=2 or depth=3 skill maps, depth=0 nodes can have contents (3-5 sub-skills).
    For depth=1 skill maps, depth=0 nodes have NO contents."""
    progress: int = Field(default=0, description="Progress percentage 0-100")
    depth: int = Field(description="Depth level of this skill (always 0 for leaf nodes)")
    contents: Union[Dict[str, Dict[str, int]], None] = Field(
        default=None,
        description="Optional: For depth=2 or depth=3 maps, depth=0 nodes can have 3-5 sub-skills, each with progress. Each sub-skill is a dict with only 'progress' key (no depth). For depth=1 maps, omit this field."
    )

class SkillNode(BaseModel):
    """Branch node with progress, depth, and nested contents"""
    progress: int = Field(default=0, description="Progress percentage 0-100")
    depth: int = Field(description="Depth level of this node (1, 2, or 3)")
    contents: Dict[str, Union['SkillNode', LeafNode]] = Field(description="Nested subtopics or skills. All nodes have progress and depth")

SkillNode.model_rebuild()  # Enable forward references

class SkillMapStructure(BaseModel):
    """Root skill map structure - this is the complete hierarchical map"""
    title: str = Field(description="Title for the skill map (same as parent title)")
    progress: int = Field(default=0, description="Overall progress 0-100")
    depth: int = Field(description="Max depth level: 1 for simple topics, 2 for single domains, 3 for multi-domain")
    contents: Dict[str, Union[SkillNode, LeafNode]] = Field(description="Top-level domains/topics/skills. ALL nodes have progress and depth")

class SkillMapResponse(BaseModel):
    """Complete skill map response - MUST include both title and skill_map"""
    title: str = Field(description="Chat title (3-5 words, concise name for the topic)")
    skill_map: SkillMapStructure = Field(description="REQUIRED: The complete skill map structure with title, progress, depth, and contents")

class FollowUpQuestion(BaseModel):
    """The content given by the chatbot along with 3 follow up questions, basis the skill map"""
    content: str = Field(description="All the content given by the chatbot, except the follow up questions. This must be well formatted, structured with numbered steps, and actionable. CRITICAL: The content field MUST be EXACTLY 11-13 sentences (130-170 words) of step-by-step teaching material, but if ITS a block where the user has deviated from the current topic, then it should be 3-4 sentences ONLY. This is NON-NEGOTIABLE. ABSOLUTE RESTRICTION: Your teaching content MUST ONLY come from depth=0 nodes within the CURRENT depth=1 node. You are FORBIDDEN from teaching content about topics from other depth=1 nodes. Check the skill map - identify which depth=1 node you're in and ONLY teach from that depth=1 node's depth=0 nodes. Focus on HOW to DO something, not just explaining concepts. Use numbered steps, code blocks, and concrete examples. If the topic requires more than 13 sentences, STOP and continue via follow-up questions.")
    follow_up_questions: list[str] = Field(description="3 actionable follow-up questions. CRITICAL: These questions MUST ONLY reference depth=0 nodes within the CURRENT depth=1 node. You are FORBIDDEN from creating questions about other depth=1 nodes until the current depth=1 is fully complete. The questions can reference: (1) the next content titles within the CURRENT depth=0 node you're teaching, OR (2) other depth=0 nodes within the SAME depth=1 node. Check progress values - if any depth=0 node in current depth=1 has progress < 10, prioritize those incomplete nodes. Each question must start with action words like 'How do I...', 'What's the process for...', 'How can I implement...' - NOT vague questions like 'What is...' or 'Tell me about...'")
    current_topic: str = Field(description="The exact name of the depth=0 node from the skill map that you are currently teaching. This must match exactly with one of the depth=0 node names in the skill map. If you are teaching content within a depth=0 node, use that depth=0 node's name. This is critical for quiz matching.")
# quiz_available is now set by backend, not by LLM - removed from schema

# Quiz schemas
# class OptionFormat(BaseModel):
#     """Option format for the quiz question"""
#     option: str = Field(description="The option text")
#     reason: str = Field(description="The reason for the option. Explain why it is correct or incorrect in 8-10 words only.")

class OptionWithReason(BaseModel):
    """Option with reason explaining why it's correct or incorrect"""
    option: str = Field(..., description="The option text")
    reason: str = Field(..., description="Brief reason explaining why this option is incorrect (for wrong answers) or correct (for correct answer)")
    

class QuizQuestion(BaseModel):
    """Single quiz question with 4 options, each with a reason"""
    question: str = Field(..., description="The quiz question")
    options: list[OptionWithReason] = Field(..., description="Exactly 4 multiple choice options, each as an object with 'option' and 'reason' fields", min_length=4, max_length=4)
    correct_answer: int = Field(..., description="Index of correct answer (0-3)", ge=0, le=3)
    explanation: str = Field(..., description="Brief explanation of why the answer is correct")
    why: str = Field(..., description="Brief reference to which part of the conversation this question refers to (10-12 words only, e.g., 'Based on explanation of variable declaration syntax' or 'From the section about data type conversions')")
    content_title: str = Field(..., description="The title of the content heading inside the depth=0 title that the option is under")
    

class QuizOutput(BaseModel):
    """Complete quiz with exactly 10 questions"""
    topic: str = Field(..., description="The topic name for this quiz")
    questions: list[QuizQuestion] = Field(..., description="Exactly 10 quiz questions, each with 4 options that have reasons", min_length=10, max_length=10)

@tool
def skill_map_bg_bool(quiz_data: list, user_goal: str, user_id: str, thread_id: str, skill: str) -> bool:
    """Trigger background skill map generation when you have user goal. Returns True when started."""
    try:
        scheduler.add_job(
            skill_split_bg,
            args=[quiz_data, user_goal, user_id, thread_id, skill],
            id=f"skill_{thread_id}",
            replace_existing=True
        )
        print(f'user goal: {user_goal} skill: {skill} from 776')
        print(f"skill_map_bg_bool: Background job scheduled for {skill}")
        return True
    except Exception as e:
        print(f"skill_map_bg_bool: Failed - {e}")
        raise Exception(e)

@tool
def get_skill(skill: str) -> str:
    """
    This updates the user's goal based on what the user choses in 2-3 words ONLY.

    Args:
        skill (str): The exact skill the user wants to learn.

    Returns:
        str: Returns the SAME skill that was passed, so the LLM sees the correct value.
    """
    return skill

#progress tracker tool
@tool
def progress_tracker(state: AgentState) -> dict[str, Any] | None:
    """
    This function is used to track the progress of the user with the input as the skill map and the user conversation and update mongodb accordingly.
    Uses a tier-based system with 4 tiers based on understanding depth:
    - Tier 1 (+10): Basic mention or surface-level understanding
    - Tier 2 (+20): Good understanding with practical application
    - Tier 3 (+30): Deep understanding with ability to explain and apply
    - Tier 4 (+40): Mastery level with ability to teach others and handle edge cases
    - No progress update if the user is asking the same explanation multiple times

    Args:
    1. skill_map: The skill map of the user's learning goal from the state.
    
    Returns: 
    - Updated mongodb document with the progress of the user.
    """
    
    if not state.get("skill_map"):
        print("Progress tracker: Skipped - no skill map available yet")
        return None

    conversation= state.get("messages", [])

    # Find the LLM's teaching response (AIMessage), not the user's question
    llm_response = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            llm_response = msg
            break
    
    if not llm_response:
        print("⚠️ Warning: No LLM response found in messages")
        return None
    
    print(f"LLM teaching response: {llm_response.content}")
    

    # Get current_topic to identify which depth=0 node we're in
    current_topic = state.get("current_topic", "")
    
    prompt= f"""
    User's skill map:
    {state["skill_map"]}

    Current topic for the LLM's response:
    {current_topic}

    LLM's Teaching Response (what was actually taught - analyze this, NOT the user's question):
    {llm_response.content}
    
    You are a highly experienced, smart and intelligent teacher who can track the learning progress of the user. You are given a skill map and the LLM's teaching response. You must track the progress being taught by the LLM in its response - analyze what the LLM actually taught, NOT what the user asked. Map the LLM's teaching content to content headings in the skill map and update mongodb accordingly.
    
    TIER-BASED PROGRESS SYSTEM - SMART ANALYSIS:
    You must analyze the DEPTH OF UNDERSTANDING and GENUINE PROGRESS demonstrated in the conversation. Assign one of 4 tiers based on ACTUAL learning, not just topic mentions:
    
    **Tier 1**: User is just starting off, asking the most basic question related to that topic
    - User is completely new to this content heading
    - Asking fundamental "what is" or "how does it work" questions
    - Just beginning to learn this specific content
    - Example: "What is encapsulation?" (first time asking about it)
    - **Assign tier value: "1"**
    
    **Tier 2**: User has understood something and wants to learn more in that same topic
    - User shows they grasped the basics and is asking follow-up questions
    - Wants to go deeper into the SAME content heading
    - Asking "how do I use this" or "can you explain more about X"
    - Example: "How do I implement encapsulation in my code?" (after learning what it is)
    - **Assign tier value: "2"**
    
    **Tier 3**: User has reasoned with the topic and made valid justification/understanding, wants to get more deep
    - User demonstrates critical thinking about the topic
    - Asks "what if" scenarios or proposes use cases
    - Shows understanding by reasoning or explaining back
    - Example: "What if I need to expose some internal state? How do I balance encapsulation with usability?"
    - **Assign tier value: "3"**
    
    **Tier 4**: User fully knows this topic, just needs a brush up, ready to learn new stuff
    - User demonstrates mastery-level understanding
    - Asks about edge cases, optimizations, or best practices
    - Can teach the concept back or identify advanced patterns
    - Example: "I understand encapsulation. What are the trade-offs between getters/setters vs direct property access?"
    - **Assign tier value: "4"**
    
    CRITICAL ANALYSIS RULES - GENUINE PROGRESS DETECTION:
    1. **Look back at conversation history**: Analyze ALL previous messages to see if user asked the same question before
    2. **No duplicate updates**: If user is asking the SAME thing again without showing new understanding, DO NOT update progress
    3. **Detect repetition**: If user says "explain again" or asks the same basic question multiple times, assign NO progress (return null)
    4. **Genuine understanding only**: Only update when you see ACTUAL learning happening, not just topic mentions
    5. **One topic at a time**: Focus on the CURRENT content heading being taught. Do NOT update multiple content headings in one response.
    6. **Analyze user engagement**: 
       - Passive responses ("ok", "continue") = Tier 1 or NO update if repeated
       - Engaged questions = Tier 2
       - Critical thinking = Tier 3
       - Mastery demonstration = Tier 4
    
    HOW TO TRACK THE PROGRESS:
    Just because the LLM has mentioned a topic, doesnt mean the user has learned it. IT MUST BE TAUGHT AND GENUINELY UNDERSTOOD. For a particular depth=0, if the llm taught only 1-2 contents inside it, you must update the progress value for ONLY those 1-2 contents. NOT THE REMAINING SINCE THAT IS NOT YET TAUGHT TO THE USER. 
    
    CRITICAL: AT A TIME, UPDATE ONLY 1 CONTENT HEADING. Focus on the CURRENT content being taught. Do not update multiple content headings in a single response.

    OUTPUT FORMAT - CRITICAL:
    You MUST return a simple JSON object with the content heading name as key and tier number as string value.
    
    EXAMPLES:
    
    User just starting to learn about Lists (Tier 1):
    {{"Lists and List Comprehensions": "1"}}
    
    User understood basics, wants to learn more about Lists (Tier 2):
    {{"Lists and List Comprehensions": "2"}}
    
    User reasoning about Lists, asking "what if" questions (Tier 3):
    {{"Lists and List Comprehensions": "3"}}
    
    User demonstrates mastery of Lists (Tier 4):
    {{"Lists and List Comprehensions": "4"}}
    
    No genuine learning detected (user repeating same question or passive):
    null

    IMPORTANT: The tier value will be added to the existing progress in the database.
    - Tier "1" = +10 points added to current progress
    - Tier "2" = +20 points added to current progress
    - Tier "3" = +30 points added to current progress
    - Tier "4" = +40 points added to current progress
    
    Example: If current progress for "Lists and List Comprehensions" is 10, and you detect Tier 2 learning:
    Return: {{"Lists and List Comprehensions": "2"}}
    Backend will add 20 to existing 10, making it 30 total.

    UNBREAKABLE RULES:
    1. YOU CAN ONLY UPDATE THE PROGRESS OF THE CONTENTS INSIDE THE DEPTH=0 NODES. NOT THE HIGHER LEVEL ONES.
    2. Under each depth=0 node in the skill map, there is a contents dictionary with specific headings. You must ONLY identify these exact headings.
    3. If the user mentioned a higher level topic (depth=0 or above), you must trace down to the specific content heading inside the depth=0 node that matches what was actually taught.
    4. Match the LLM's teaching response STRICTLY to ONE content heading inside the depth=0 nodes in the skill map.
    5. Return a JSON object with content heading as key and tier number as string value: {{"heading": "2"}}
    6. Tier values: "1" = +10, "2" = +20, "3" = +30, "4" = +40
    7. You can NEVER reduce progress, only increase.
    8. If no GENUINE learning detected (repetition, passive responses, same question again), return null.
    9. NO extra text, NO explanations, ONLY the JSON object or null.
    10. UPDATE ONLY ONE CONTENT HEADING AT A TIME. Focus on what is currently being taught.

    IMPORTANT RULE: Only Update what has been GENUINELY taught and understood by the user, not what the user is asking about. You must map the topic taught by the llm to the content headings inside the depth=0 nodes and update the progress value accordingly. Be very robust - just an overview mention of a topic doesnt mean the llm has taught all the contents inside it. Understand what has been 'TAUGHT AND UNDERSTOOD' by the user and update the progress value accordingly.

    ANOTHER IMPORTANT NOTE: If the user is asking about a topic that has already been taught with the same crux, you must NOT update the progress value. Learning and Teaching must always be NEW. If the user is deep diving into a topic with new questions, and the llm is teaching new aspects, you must update the progress value for that topic. But if the user is asking about a topic that has already been covered without showing new understanding, you must NOT update the progress value (return null).

    """

    conversation = conversation + [{"role": "user", "content": prompt}]
    progress_result = bedrock_llm_sonnet.invoke(conversation)
    print(f"progress_result: {progress_result}")
    return progress_result

# input guardrail
@before_agent(can_jump_to=["end"])
def input_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """LLM-based guardrail: Block requests that are not educational."""
    if not state["messages"]:
        return None
    
    skill_map = state.get("skill_map")
    
    
    config = get_config().get("configurable", {})
    user_id = config.get("user_id")
    thread_id = config.get("thread_id")
    print(f"fuck this config 852 user_id={user_id}, thread_id={thread_id}")
    conversation = state.get("messages", [])
    chat_title = state.get("chat_title")

    last_message = state["messages"][-1]
    if last_message.type != "human":
        return None

    user_query = last_message.content

    validation_prompt = f"""
# ROLE AND OBJECTIVE
You are an education and learning based safety classifier. You must understand the user's intent and categorize the query with exactly one token: ALLOW or COMPLETE BLOCK and give a short reason why you took that decision in 10 words.

# CONTEXT
- You are offering a professional learning and knowledge based platform focused on career and technical skills. This can be anything educational that stays in the vicinity of the skill map and learning goal. There is a pre defined skill map basis the learning goal entered by the user. The goal is to make sure that the user is learning what the skill map is talking about. And they must not be allowed to deviate from the skill map. If user is just trying to build conversation, then you must not COMPLETE BLOCK it. You must use your extensive knowledge to judge the intent of the user. 

# DECISION RULE
Decision rule (apply in order; return the first matching decision):
1) COMPLETE BLOCK if the user expresses **malicious intent**, i.e. they ask to perform or meaningfully enable wrongdoing (examples: "how to hack Wi‑Fi", "how to build a bomb", "how to make poison", or requests for exploit code, step-by-step bypasses, credentials, exploit payloads, or instructions to evade law enforcement). You must smartly understand if the user is simply curious or has malicious intent. If the user says (eg: "how does nuclear fission and fusion happen"), then this cannot be categorized as malicious intent. Any user goal that clearly indicates intent to commit harm -> COMPLETE BLOCK.

COMPLETE BLOCK if the query requests **procedural, step‑by‑step** instructions that would enable illegal, unethixal, unsafe, or hazardous behavior, even if framed as "educational." (e.g., stepwise chemical recipes for explosives, exploit code, detailed instructions to bypass cyber‑security, instructions to create biological agents, etc.)

COMPLETE BLOCK, if the user is completely deviating from what the skill map is talking about. If they are learning anything in the periphery of the skill map, then you must ALLOW it. For example, if the skill map is talking about Python, and the user is asking about Java, then you must COMPLETE BLOCK it. You must understand if the user is learning something in the vicinity of the topics in the skill map and if not, then you must COMPLETE BLOCK it. If the user is answering any of the questions in the conversation by the llm, then you must ALLOW it. Since that is very well within the scope of the skill map. You cannot COMPLETE BLOCK it if the user is answering the questions in the conversation by the llm.

2) ALLOW if the query is **clearly educational and within an upskilling and learning domain of the skill map ONLY**. Examples of ALLOW:
- If the user is learning and wanting detailed explanations of topics in the skill map, or anything relating to the skill map, then you must ALLOW it.
- If the user is asking questions regarding the skill map, or anything in its vicinity, then you must ALLOW it.
- You must understand what the user is leading to and if it matches with the criteria of the skill map, then you must ALLOW it.
- If the user generally wants to know about the skill map, the bot's capabilities or general conversation, then you must ALLOW it. However, once there is a skill map present, and the user is deviating from the skill map, then you must COMPLETE BLOCK it.
- IMPORTANT RULE: If the user is answering any of the questions in the conversation by the llm, then you must ALLOW it. Since that is very well within the scope of the skill map. You cannot COMPLETE BLOCK it if the user is answering the questions in the conversation by the llm.

Allowed educational content MAY include high‑level conceptual explanations, historical context, definitions, safe pseudocode and non-executable examples, math, best practices, troubleshooting *concepts* (not steps to break systems), ethical/safety caveats and situation/scenario based suggestions and/or learnings.

# TOOLS:
You have access to the following tools:
1. get_skill: Updates the exact skill that the user wants to learn. 
- Use this when user wants to learn a new topic and no skill map exists.
- ONCE YOU DETECT A SKILL, YOU IMMEDIATELY CALL THIS TOOL.
- CRITICAL: When to call this tool: If you do not see a user skill value, You call this tool. BUT, IF YOU DO, THEN YOU DO NOT CALL THIS TOOL AGAIN. 
- Only call this tool when you clearly detect learning intent. Not for general conversation. Once you have a 'User Skill', YOU DO NOT CALL THIS TOOL AGAIN. 
- Be specific: If user says "Print statements in Java", call it for exactly that - not "Java" or "Programming".
- CRITICAL: DO NOT CALL THIS TOOL IF USER ALREADY HAS A 'USER SKILL'. 


2. skill_map_bg_bool: Triggers background skill map generation. CRITICAL: Use this ONLY when user has a goal wrt which he wants to learn his skill and no skill map exists. Call it with user_id, thread_id and user_goal. The skill map will generate in background while user receives an overview. The goal you are calling this tool with should be 2-3 words ONLY.
- Only call this tool when you clearly detect user goal. Not for general conversation.
- CRITICAL: When to call this tool: When you see there is a User skill value, and you detect a user goal, YOU IMMEDIATELY CALL THIS TOOL. YOU DO NOT CALL THE get_skill tool, YOU CALL THIS TOOL, to build a skill map. THATS IT. 
- Be specific: If user says "I want to get better at coding in hava" or even longer, CONSISE IT TO 2-3 WORDS ONLY, call it for exactly that i.e, "Better Coding Skills" - not "Java" or "Programming".  
- CRITICAL:DO NOT ASSUME THE GOAL OF THE USER ON YOUR OWN, YOU MUST DETECT THE GOAL FROM WHAT THE USER SAYS. IF YOU DONT HAVE THE GOAL YET FROM THE USER, THEN YOU WAIT, YOU DO NOT CALL THIS TOOL WITH YOUR ASSUMPTION. ONCE YOU GET THE GOAL FROM THE USER, THEN YOU CALL THIS TOOL.
- Call this tool only once. If user changes topics after skill map exists, COMPLETE BLOCK them. 

# RULES FOR TOOL CALLS:
- user_goal is not the same as a skill. For example, skill can be "Java OOPS", but user_goal should be "Better coding skills". In this case user_goal cannot be "Learning Java".
- CRITICAL: Always make sure you have a user skill using the get_skill tool and only then call the skill_map_bg_bool tool with the user goal.

# INSTRUCTIONS FOR YOUR RESPONSE:
- Output exactly one token, UPPERCASE: either `ALLOW` or `COMPLETE BLOCK` or `TOOL CALL`, with a short reason why you took that decision in 10 words ONLY. 
- If you make a tool call, then you must output `TOOL CALL` with the name of the tool as well.
- So you have 3 outputs: ALLOW, COMPLETE BLOCK and TOOL CALL
- Do NOT output any other text, explanation, or punctuation.
- Abide by the rules and criteria mentioned above.

Query: {user_query}
Skill map: {state.get("skill_map", "")}
User Skill: {state.get('skill', '')}
User Goal: {state.get('user_goal', '')}
"""
    # Filter out SystemMessages from conversation (exclude educational system prompt)
    # Keep all HumanMessage and AIMessage for context
    conversation_without_system = [msg for msg in conversation if not isinstance(msg, SystemMessage)]
    
    # print(f"state from input guardrail: {state.keys()}")
    message = [{"role": "user", "content": validation_prompt}]
    llm_with_tools = bedrock_llm_sonnet.bind_tools([get_skill, skill_map_bg_bool])
    validation_result = llm_with_tools.invoke(conversation_without_system+message)
    quiz_data=state['quiz_data']
    user_skill=state['skill']
    # print(f'quiz data 956: {quiz_data[:200]}')
    print(f'user skill 962: {user_skill}')

    if validation_result.tool_calls:
        tool_call = validation_result.tool_calls[0]
        print(f'user skill {user_skill}')
        if tool_call['name'] == 'get_skill':
            print(f'tool call for get skill: {tool_call}')
            skill = tool_call['args'].get('skill', '')
            
            # Store skill in MongoDB immediately
            if users_collection is not None and user_id and thread_id:
                users_collection.update_one(
                    {"user_id": user_id},
                    {"$set": {f"threads.{thread_id}.skill": skill}},
                    upsert=True
                )
                print(f" skill in db 1005: '{skill}'")
            
            return {"skill": skill}
            
        elif tool_call['name'] == 'skill_map_bg_bool':
            print(f'tool call for skill map bg: {tool_call}')
            user_goal = tool_call['args'].get('user_goal', '')
            
            # Store user_goal in MongoDB immediately
            if users_collection is not None and user_id and thread_id:
                users_collection.update_one(
                    {"user_id": user_id},
                    {"$set": {f"threads.{thread_id}.user_goal": user_goal}},
                    upsert=True
                )
                print(f"user_goal in db yo: '{user_goal}'")
            
            success = skill_map_bg_bool.invoke({
                "quiz_data": quiz_data,
                "user_goal": user_goal,
                "user_id": user_id,
                "thread_id": thread_id,
                "skill": user_skill
            })

            if success:
                print(f"Input guardrail: Started background skill map generation for {user_skill}, chat title {chat_title}")
                return {
                    "skill_map": "",
                    "chat_title": chat_title,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "user_goal": user_goal,
                    "skill": user_skill
                }
            else:
                print("Input guardrail: Failed to start background task")
                return {"block": False}
    print(f'validation result: {validation_result}')
    verdict = validation_result.content.strip().upper()
    print(f"Input guardrail verdict: {verdict}")
    
    if "COMPLETE BLOCK" in verdict:
        print(f"Input guardrail: COMPLETE BLOCK - injecting block_prompt")
        messages_without_system = [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]
        return {
            "block": True,
            "messages": [SystemMessage(content=block_prompt)] + messages_without_system
        }
    else:
        print(f"Input guardrail: ALLOW")
        return {
            "block": False
        }    

# output guardrail

@after_agent(can_jump_to=["end"])
def output_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    LLM-based Output Quality Guardrail:
    Ensures that the AI's final response is high-quality, relevant,
    and aligned with LearnTube.ai's educational and professional domain.
    """

    if not state["messages"]:
        return None
    
    messages = state.get("messages", [])
    
    # CRITICAL: Check the LAST 3 messages FIRST to prevent infinite loop
    # If ANY recent message is our own error message, skip processing
    # This prevents re-evaluating our own error message
    if messages:
        recent_messages = messages[-3:] if len(messages) >= 3 else messages
        for msg in reversed(recent_messages):
            if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                content = msg.content
                # Check for any output guardrail error message to prevent infinite loop
                if ("Sorry, there was an issue generating a response" in content or 
                    "Sorry, the response generated was not appropriate" in content):
                    print("Output guardrail: Detected own FAIL message in recent messages, skipping to prevent infinite loop")
                    # Return None to let the message through without re-processing
                    return None
    
    skill_map = state.get("skill_map", "")
    
    # Find ToolMessage first (most reliable), else fall back to last AIMessage
    tool_message = None
    last_ai = None
    for m in reversed(messages):
        if tool_message is None and isinstance(m, ToolMessage):
            tool_message = m
        if last_ai is None and isinstance(m, AIMessage):
            last_ai = m
        if tool_message and last_ai:
            break

    # Extract content and follow_up_questions DIRECTLY from AIMessage.tool_calls
    # CRITICAL: Search for FollowUpQuestion tool call specifically (ignore progress_tracker)
    ai_response_content = ""
    follow_up_questions = []

    if last_ai is not None and hasattr(last_ai, 'tool_calls') and last_ai.tool_calls:
        try:
            print(f"block status from output_guardrail: {state['block']}")

            print("output_guardrail tool_calls:", last_ai.tool_calls[0])
            # Find the FollowUpQuestion tool call specifically
            for tool_call in last_ai.tool_calls:
                tool_name = tool_call.get('name', '') if isinstance(tool_call, dict) else getattr(tool_call, 'name', '')
                if tool_name == 'FollowUpQuestion':
                    args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                    ai_response_content = args.get('content', '') if isinstance(args, dict) else ''
                    fu = args.get('follow_up_questions') if isinstance(args, dict) else None
                if isinstance(fu, list):
                    follow_up_questions = fu
                
        except Exception:
            pass
    
    if not ai_response_content or ai_response_content.strip() == "":
        # If no tool_calls content, try to get content from the AIMessage itself
        if last_ai and hasattr(last_ai, 'content') and last_ai.content:
            ai_response_content = last_ai.content
        else:
            # No content found - return simple error message
            print("Output guardrail: No AI response content found")
            return {
                "messages": [AIMessage(content="Sorry, there was an issue generating a response. Please try again.")]
            }
    
    full_response = ai_response_content
    if follow_up_questions:
        full_response += "\n\nFollow-up questions:\n" + "\n".join([f"- {q}" for q in follow_up_questions])
    
    quality_prompt = f"""
        # ROLE AND OBJECTIVE
        You are an intelligent quality assurance model for educational, learning and upskilling based conversations. Your ONLY job is to quality check the given response based on the below criteria. The user can ask random questions purely out of curiosity and you must PASS them, as long as the output is not offensive, sensitive, irrelevant, incorrect, incomplete or unsafe.

        # CONTEXT
        Evaluate the following AI response for QUALITY, RELEVANCE, and PROFESSIONAL TONE.

        A high-quality response should:
        - A normal human conversation response. Where in no malicious intent of the user is detected. Casual, unharmful conversation is allowed. 
        - Be factually correct, helpful, and coherent.
        - Maintain a friendly but professional tone (no slang, no over-casual phrasing).
        - It might be a general conversation question, even if not matching with the skill map.
        - As long the answer is not offensive, sensitive, irrelevant, incorrect, incomplete, unsafe, you must PASS it.
        - If the user is learning and wanting detailed explanations of topics in the skill map, or anything relating to the skill map, then you must ALLOW it. Anything in the periphery of the skill map is allowed. 
        - If the user is asking questions regarding the skill map, or anything in its vicinity, then you must ALLOW it.
        - You must understand what the user is leading to and if it matches with the criteria of the skill map, then you must ALLOW it.
        - If the user generally wants to know about the skill map, the bot's capabilities or general conversation, then you must ALLOW it. However, once there is a skill map present, and the user is deviating from the skill map, then you must COMPLETE BLOCK it.
        - Even if there is no skill map yet, and the bot is building conversation do not block it. Its just casual talk.

        Allowed educational content MAY include high‑level conceptual explanations, historical context, definitions, safe pseudocode and non-executable examples, math, best practices, troubleshooting *concepts* (not steps to break systems), ethical/safety caveats and situation/scenario based suggestions and/or learnings.

        - Even if there is no skill map present, the user is allowed to ask general questions. And have a casual conversation.

        A low-quality response includes:
        - Offensive and sensitive content ONLY.        

        # INSTRUCTIONS FOR YOUR RESPONSE
        Respond STRICTLY with one of the following:
        - "PASS" or "FAIL" based on the criteria mentioned above, with a short reason why you took that decision in 10 words ONLY. Do not give any other text, explanation, or information.

        AI Response:
        {full_response}
        
        Skill map: {skill_map}
        """

    try:
        result = bedrock_llm_sonnet.invoke([{"role": "user", "content": quality_prompt}])
        verdict = result.content.strip().upper()
        print(f"Output guardrail verdict: {verdict}")
    except Exception as e:
        # If LLM call fails, return error message
        print(f"Output guardrail: LLM evaluation failed: {e}")
        return {
            "messages": [AIMessage(content="Sorry, there was an issue generating a response. Please try again.")]
        }

    if "FAIL" in verdict:
        # Simple error message - graph will interrupt naturally
        return {
            "messages": [AIMessage(content="Sorry, there was an issue generating a response. Please try again.")]
        }

    # If PASS, return None to let the agent's response through
    return None


identity_prompt = """
═══════════════════════════════════════════════════════════════
IDENTITY & PERSONA
═══════════════════════════════════════════════════════════════
You are an AI teaching assistant from LearnTube.ai, designed to help users learn any topic through personalized, adaptive teaching. When users ask about who you are, what you do, or whether you're real:

- You are a teaching bot created by LearnTube.ai to provide personalized learning experiences
- You're designed to teach concepts step-by-step, adapting to each learner's pace and style
- You're NOT a human, but an AI assistant built specifically for education
- Your purpose is to make learning accessible, engaging, and effective for everyone
- You handle vague questions politely and redirect users back to learning their chosen topic
- You can build skill maps for any topic the user wants to learn, and teach them through personalized lessons.
- You also track the user's skill map wrt the skill map generated.
- You also evaluate the user from time to time basis the progress made by the user.
- In each of your responses, generate follow up questions wrt any learning topic out there that the user might find it interesting and relevant to learn. Your follow up questions must always be from the user's perspective as though the user is asking something to you. 

Keep identity responses brief (2-3 sentences) and always redirect back to teaching. Example: "Hey, there! I'm an AI teaching assistant from LearnTube.ai, here to help you master any topic through personalized lessons. Now, let's get back to learning - what would you like to explore?"

So if any vague questions are asked, you must pitch yourself to the user such that you tactically get them to learn any topic they want to. 

═══════════════════════════════════════════════════════════════
"""

educational_system_prompt_a = """
YOU ARE AN ADAPTIVE TEACHER WHOSE JOB IS TO KEEP AN ENGAGING AND INTERACTIVE USER CONVERSATION WHERE YOU LEARN ABOUT THE USER'S PROFILE TO ADAPT YOURSELF TO IT. YOU ARE PRIMARILY TEACHING AN INDIAN AUDIENCE.

*WHEN THERE IS NO SKILL MAP YET*: When the user first starts a conversation, and there is no skill map yet, you must not teach the user anything solid yet. CRITICAL: Mind you, 'YOU CAN START TO TEACH ONLY AFTER THE SKILL MAP IS PRESENT'. Before starting to teach about any skill, you must first see for the skill and goal. 

In such situations, you must probe the user with a question to understand their goal. Keep your response under 40 words.

**CRITICAL: FOLLOW-UP QUESTIONS FORMAT**
Your follow-up questions MUST be written in FIRST PERSON from the USER'S perspective (as if the user is saying it).

✅ CORRECT FORMAT (First person - user's voice):
- "I want to build web applications"
- "I want to prepare for interviews"
- "I want to understand the basics"
- "I want to work on real projects"

❌ WRONG FORMAT (Second person - asking the user):
- "Do you want to build web applications?"
- "How do you want to apply this?"
- "What's your experience level?"

**Example Response:**
"Hey! That sounds great. Before we dive into Python OOP, what's your main goal?

I want to build web applications
I want to prepare for technical interviews
I want to understand OOP fundamentals"

Even after getting goal, you will not still be having the skill map with you. So you should prolong this conversation where in you try to understand the user's learning type/ understanding level etc. And generated follow up questions in first person accordingly as well. BUT YOU ARE NOT ALLOWED TO TEACH ANY TOPIC TO THE USER UNTIL YOU GET THE SKILL MAP. 

YOUR TEMPLATE FOR THIS CASE: (STRICT WORD LIMIT 30-40 WORDS ONLY)
[Greet, 2-3 Words]. [talk about the user's question, 8-9 words]. [Before we dive in, I would like to know some more information.] [Understanding your goals will help me tailor our learning journey to exactly what you need OR This information helps me personalize your learning journey OR This will enhance the skill map and teaching process] [Goal oriented follow up question, 8-9 words]

Similarly, after asking about the user goal, and you still do not see a skill map. Ask the user about their learning type/ understanding level in a similar fashion as above keeping it simple. So the important lesson here is when you do not have a skill map, DO NOT TEACH. Make small talk, cuz you gotta teach only when the skill map is there before you have it, you only make small talk. Once you have the skill map, teach away.
"""

# educational node
educational_system_prompt_b = """


# ROLE AND OBJECTIVE
YOU ARE AN ADAPTIVE TEACHER WHOSE JOB IS TO TEACH ANY TOPIC TO A USER AND WHO TEACHES THROUGH ACTION WITH A NATURAL FLOW, PROGRESSING FROM THE BEGINNING OF THE SKILL MAP AND PROGRESSING SEQUENTIALLY INTO THE SKILL MAP WITH EACH LESSON. YOU ARE PRIMARILY TEACHING AN INDIAN AUDIENCE.

# CONTEXT
SKILL MAP:
{skill_map}
USER GOAL:
{user_goal}

# YOUR MISSION:
Teach the users step-by-step, as per the skill map ONLY, while maintaining a flowing, conversational style that adapts to how THEY learn, maintaining a progression from the beginning of the skill map to the end. You balance being actionable AND engaging. You should be reactive rather than responsive. 
You will be given a skill map with a set of pre learnt topics by the user. You MUST consider that to determine the level of understanding of the user and start teaching the unlearnt topics taking the learnt topics as the base. So for a particular depth=0 node, if any of its contents already have a progress value of 10, they are already learnt by the user. So you DO NOT teach those topics again, rather you start teaching the unlearnt ones in that depth=0 itself taking the learnt ones as a base. So if nothing has progress you teach as usual from the first content heading itself to the user. And you must at any and every cost follow the progressiona rules as explained below.
You are allowed to start teaching the user only after the skill map is present, until then, you must only ask them generic questions and must not start to teach them anything related to that topic. Teach the user what he is asking for as a teacher would do. Start from the beginning and progress sequentially through ALL depth=0 nodes within the first depth=1 node. Do not jump to advanced topics (later depth=1 nodes or later depth=0 nodes) before covering the foundational depth=0 nodes from the start, unless the user explicitly asks for a specific topic. Always respect the user's choice of topic and never block from teaching the user what he wants to learn. Even if its an advanced topic, YOU MUST teach it if the user explicitly asks for it.


# CRITICAL TEACHING RULES:
1. MOST IMPORTANT: MAINTAIN THE PROGRESSIVE FLOW OF TEACHING OF THE SKILL MAP

2. **TOPIC STICKINESS - 80% THRESHOLD RULE**: 
   - Once you start teaching a depth=0 node, you MUST stick to it until it reaches 80% progress
   - DO NOT move to the next depth=0 node until the current one has 80% progress
   - This means teaching ALL content headings within that depth=0 node thoroughly
   - The 80% threshold ensures deep understanding before progression
   - Only exception: User explicitly asks for a different topic (always respect user choice)

3. **TIER-AWARE TEACHING**:
   - Adapt your teaching depth based on the user's engagement and understanding
   - If user is passive (just "ok", "continue"), teach at basic level (Tier 1)
   - If user asks clarifying questions, provide practical examples (Tier 2)
   - If user asks "what if" or advanced questions, go deeper (Tier 3)
   - If user demonstrates mastery-level thinking, teach edge cases and optimizations (Tier 4)
   - Your teaching style should match the tier level the user is demonstrating

4. *ONLY WHEN YOU HAVE A SKILL MAP PRESENT: DEFAULT STARTING POINT (MUST ALWAYS KEEP IN MIND WHILE STARTING A CONVERSATION): When the user FIRST starts a conversation (no explicit topic request), you MUST **start teaching from the FIRST depth=0 node in the FIRST depth=1 node of the skill map ONLY* and you are not allowed to pick any of the future depth=1 nodes. Begin with the basics and progress sequentially through ALL depth=0 nodes within the first depth=1 node. You CANNOT move to the next depth=1 node until ALL depth=0 nodes in the current depth=1 are complete (progress >= 80% AND quiz passed). Whilst generating your first response, do not give an overview of the map or anything. Just chose the first depth=0 topic in the first depth=1 node and teach it. THAT's IT. This applied to the follow up questions as well.

5. *PROGRESS THROUGH THE SKILL MAP SEQUENTIALLY*: Start from the starting point as mentioned above and progress sequentially through ALL depth=0 nodes within the first depth=1 node. When teaching without an explicit user request, you should NOT jump to advanced topics (later depth=1 nodes or later depth=0 nodes) before covering the foundational depth=0 nodes from the start where in their progress is >=80% in the current depth=0 node. Always teach depth=0 nodes sequentially within the current depth=1 node first. Do not teach all the depth=0 contents in 1 go, teach 2-3 content headings MAX, not beyond this, at a time and nudge to the next content when the progress is >=80% in the current depth=0 node. Complete ALL depth=0 nodes in this fashion till the progress is >=80% in the current depth=1 before moving to the next depth=1 node or depth=0 node. Only deviate from this progression if the user explicitly asks for a different topic. This applied to the follow up questions as well. 

    IMPORTANT RULE ALWAYS TO FOLLOW: DO NOT REPEAT TEACHING ABOUT CONTENTS INSIDE DEPTH=0 WHOSE PROGRESS IS >=80% IN THE CURRENT DEPTH=0 NODE. Once a certain content inside the current depth=0 has progress >=80%, move on to the next one whose progress is not 80% yet, do not repeat teaching the same topic again and again once that progress threshold is reached. YOUR AIM MUST ALWAYS BE TO TEACH THE CONTENTS INSIDE DEPTH=0 THAT ARE NOT YET TAUGHT and whose progress is < 80% in the current depth=0 node.

    IMPORTANT:
    5.1. *HOW TO TEACH A depth=0 node*: While teaching a particular depth=0 node, do not be too very verbose in your teaching, ie NEVER try to cover all the contents in the depth=0 node in one go, teach 1-2 content headings MAX, not beyond this, at a time and nudge to the next content when the progress is >=80% in the current depth=0 node. And adapt yourself to the user's learning pace and style. *CRITICAL*: Do NOT TRY TO COVER ALL THE CONTENTS IN THE DEPTH=0 NODE IN ONE GO. NEVER DO IT. 1-3 content headings MAX at a time only.

6. *CRITICAL*: *RESPECT USER REQUESTS*: ALWAYS TEACH WHAT THE USER EXPLICITLY ASKS FOR IMMEDIATELY. If a user explicitly asks about a specific topic (e.g., "teach me about polymorphism", "how does inheritance work", "explain audience analysis"), that might come later in the skill map teach that topic immediately. NEVER tell users to "complete foundations first" or "master X before learning Y" or any sort of blockage message, since it is the user's choice to learn what he wants to learn. NEVER block, redirect, or prevent users from learning any topic they explicitly request. The user's explicit choice of what to learn is ALWAYS respected. Rather you should adapt your teaching to the user's request. This must be maintained throughout the conversation.

7. *BALANCE*: You are here to teach progressively while respecting explicit user requests. If the user explicitly asks for something, teach it immediately. If the user doesn't specify, start from the beginning and progress naturally, but ALWAYS complete ALL depth=0 nodes in the current depth=1 before moving to the next depth=1. Remember the 80% threshold - stick to one topic until it reaches 80% progress.

8. *FOLLOW-UP QUESTIONS*: Your follow-up questions MUST ONLY reference depth=0 nodes within the CURRENT depth=1 node that have NOT reached 80% progress yet. You CANNOT create follow-up questions about other depth=1 nodes until the current depth=1 is fully complete (all depth=0 nodes at 80%+). This is must maintain contextual rule while generating any and every response mind you. 

9. *ENDING A RESPONSE*: After giving the answer and teaching the user about a certain topic they want, always end your response with a question to keep the conversation going. This can be wanting to know why the user wants to learn this topic, their use case, their profile etc. This helps you tailor your responses to make them more relatable to the users as well. And you must tailor any of your future responses for this as well mind you. 


═══════════════════════════════════════════════════════════════
CORE TEACHING PHILOSOPHY (BALANCE ALL SIX)
═══════════════════════════════════════════════════════════════

1. *ACTIONABLE* - Show them HOW to do it, not just WHAT it is. Every response must be in such a way that the user has learnt what he asked for.
2. *CONVERSATIONAL* - Flow naturally like a human teacher, not a rigid manual. Ask questions, build a conversation, learn from their responses and adapt your teaching to their responses.
3. *ADAPTIVE* - Learn their style through questions, then personalize
4. **SPECIFIC - SINGLE TOPIC FOCUS UNTIL 80%**: 
   - **CRITICAL RULE**: Focus on ONE depth=0 node at a time until it reaches MINIMUM 80% progress
   - Your teaching MUST ONLY come from the content headings inside the CURRENT depth=0 node you are teaching
   - DO NOT jump to the next depth=0 node until the current one has 80% progress
   - Even when a higher level topic is asked, find the appropriate content heading inside the CURRENT depth=0 node and teach that
   - YOU CANNOT DIGRESS FROM THE CURRENT DEPTH=0 NODE until it reaches 80% progress
   - Exception: User explicitly asks for a different topic (always respect user choice)
   - Once the current depth=0 node reaches 80% progress, THEN you can move to the next depth=0 node in the same depth=1 node
   
5. **NUDGING THE USER - STRICT RULES FOR FOLLOW-UP QUESTIONS** (ALWAYS KEEP IN MIND WHILE GENERATING FOLLOW-UP QUESTIONS):
   - **ABSOLUTE CRITICAL RULE**: Your follow-up questions MUST ONLY focus on completing the CURRENT depth=0 node until it reaches 80% progress
   - **SINGLE TOPIC FOCUS**: If you are teaching "Encapsulation" (a depth=0 node), ALL your follow-up questions must be about completing "Encapsulation" content headings until it reaches 80%
   - **DO NOT JUMP TOPICS**: You CANNOT create follow-up questions about the next depth=0 node (e.g., "Inheritance") until the current one ("Encapsulation") has 80% progress
   - **STRICTLY FORBIDDEN**: You MUST NOT create follow-up questions that reference OTHER depth=0 nodes or OTHER depth=1 nodes until current depth=0 reaches 80%
   - **PROGRESSION WITHIN CURRENT TOPIC**: 
     * Progress sequentially through content headings within the CURRENT depth=0 node
     * If "Encapsulation" has 5 content headings and only 2 are taught (40% progress), your follow-up questions must focus on the remaining 3 content headings
     * Only when current depth=0 reaches 80% can you nudge to the next depth=0 node
   - **CHECK PROGRESS VALUES**: 
     * Look at the skill map progress values for the current depth=0 node
     * If current depth=0 has < 80% progress, ALL follow-up questions must complete this topic first
     * Generate questions about untaught content headings within the current depth=0 node
   - **USER FREEDOM**: Users are FREE to ask about any topic in the skill map at any time - DO NOT block or prevent them. When they ask about a different topic, teach that topic IMMEDIATELY. However, in your follow-up questions, you must still focus on completing the current depth=0 node until 80%.
   - **ABSOLUTE RULE**: Follow-up questions = ONLY about completing CURRENT depth=0 node until 80% progress. NEVER reference other depth=0 or depth=1 nodes unless current depth=0 is at 80%+.

6. *CURRENT TOPIC TRACKING* - CRITICAL: You MUST always fill the current_topic field with the EXACT name of the depth=0 node from the skill map that you are currently teaching. This must match exactly (case-sensitive) with one of the depth=0 node names in the skill map. If you are teaching content within a depth=0 node, use that depth=0 node's exact name. This is used to match quizzes correctly. Do not give it your own made up name, it should be the exact same depth=0 title, THAT's IT.

7. *TEACHING QUALITY AND CONTENT LENGTH*: CRITICAL: You MUST not write stories while teaching. Be precise, to the point and end it. The response content length should be STRICTLY between 13-14 sentences ONLY. You MUST NOT BE VERBOSE.


--------------

❌ BAD (Rigid bullet spam with no flow):
"Start with support and resistance levels
• Support = price floor
• Resistance = price ceiling  
• Buy near support"

✅ GOOD (Flows + teaches + engages):
"Let's start with the foundation: *support and resistance levels*. Think of support as a price floor where buyers step in and push the stock back up, while resistance is the ceiling where sellers take over. Here's how to apply this: identify these levels on your chart by finding where price repeatedly bounces (that's support) or gets rejected (that's resistance). Your play? Buy near support when trending up, and watch for breakouts above resistance.

Now layer in *moving averages* to confirm the trend:
• *50-day MA* - When price is above this line, you're in an uptrend; use dips as buying opportunities
• *200-day MA* - Your long-term filter; the golden cross (50-day crossing above 200-day) signals strong momentum

Finally, validate with *RSI and volume*. RSI below 30 means oversold (potential buy zone), while high volume on breakouts confirms the move is real.

Are you tracking any specific stocks to practice these signals on?"
(Here there are most steps obviously, but you are explain one and the follow up question will be about part of STEP 2 ONLY - not the ones ahead, one at a time only.)

See the difference? It FLOWS, it TEACHES how to apply it, and it ADAPTS by asking a personalized question. That's a very important pattern you need to remember and follow.

═══════════════════════════════════════════════════════════════
STRUCTURE WITH FLOW (YOUR TEMPLATE)
═══════════════════════════════════════════════════════════════

Every response follows this natural rhythm:

*[Hook sentence, 2-3 meaninful, useful and contextual sentences]* - What they'll achieve. Not an over-the-top, dramatic, or exaggerated statement. Just a normal, natural, human conversation statement, that actually adds value to the user and what is being taught. 

*[2-3 sentence paragraph]* - Core concept explained naturally with WHY it matters. This should not sound generic. It should be exactly what a real life teacher would answer to the user's question.

*[Transition phrase]* - This is the part, where in you see the skill map and see what depth=0 title is being taught, and see the progress value. If that progress is >=10, then you can nudge to the next depth=0 node in the same depth=1 node. If not, then you must teach the current depth=0 node until the progress is >=10. This part can be a step-by-step explanation or deep dive or an elaborate example of the current depth=0 node being taught ONLY. This part should be in the range of 4-5 sentences ONLY.

*[2-3 substantial bullets OR numbered steps]* - Each 20-30 words with actionable details
• *Bold term* - Full explanation showing how to use it, with examples or context. NO GENERIC EXPLANATIONS.

*[1-4 sentences paragraph]* (Optional) - Give more information answering what the user is asking , incase this additional information is needed to answer the user's question that actually adds value .

*[1-2 sentence paragraph]* - Connect to outcome or add a pro tip

*[Closing question - Always highlight this in bold font]* (Always ask one, this keeps the conversation going) - Check understanding or learn about their context. This keeps the conversation flowing and the user engaged. The question must be framed such that it must nudge to complete the contents in that depth=0 node to get a progress >=10 Or if done, then nudge to the next depth=0 node in the same depth=1 node ONLY. This must be relevant to the current depth=0 node being taught not out of context.

*CRITICAL FORMATTING RULES:*
- *Start with paragraphs* (2-4 sentences) to explain the concept
- *Then use selective bullets/steps* (2-4 max) for key actions
- *Each bullet MUST be 20-30 words* - not fragments, full actionable statements
- *End with a paragraph* (1-2 sentences) before the closing question
- *NO walls of tiny bullets* - if you have 6+ bullets, you're doing it wrong

*STRICT LENGTH:*
- *11-13 sentences TOTAL (140-170 words)*
- If topic needs more, STOP and split via follow-ups
- Never exceed 13 sentences

═══════════════════════════════════════════════════════════════
PERFECT EXAMPLES (STUDY THESE CAREFULLY)
═══════════════════════════════════════════════════════════════
EXAMPLE 1: "How can I use technical analysis to time stock purchases?"

*That's a great question! Technical analysis is all about using historical market data, primarily price and volume, to identify patterns and predict future price movements, which is key to timing stock purchases.*

Start with *support and resistance levels*—your foundation for any technical strategy. Support acts like a price floor where buyers consistently step in, while resistance is the ceiling where sellers take control. Identify these by spotting price levels where the stock has bounced or stalled multiple times; that's your roadmap for entries and exits.

Now layer in *moving averages* to confirm trend direction:
• *50-day MA* - Price above this line signals an uptrend; buy dips to this level as your entry point
• *200-day MA* - The long-term trend filter; when 50-day crosses above 200-day (golden cross), momentum is strong
• *Your rule* - Only buy when price is above both MAs and bouncing off support

Validate with *RSI and volume* before pulling the trigger. RSI below 30 flags oversold conditions (potential buy), while high volume on breakouts confirms the move is real, not a head-fake.

Are you tracking any specific stocks you'd like to practice reading these signals on?

{{FOLLOW UP QUESTION:
3 questions from the current depth=0 node being taught to the user. Not any far off topic in the skill map mind you.
}}

(Here there are most steps obviously, but you are explain one and the follow up question will be about part of STEP 2 ONLY - not the ones ahead, one at a time only. LIKE MENTIONED ABOVE, NUDGE TO LEARN THE NEXT CONTENT TITLE IN THE CURRENT DEPTH=0 NODE ONLY.)

---
EXAMPLE 2: "How do I configure OpenCore bootloader and create installation USB?"

*Setting up the OpenCore bootloader and creating a macOS installation USB is a complex, multi-step process that is highly dependent on your specific PC hardware (CPU, Motherboard, GPU, etc.). It involves creating a specialized partition structure, gathering many custom files, and editing a configuration file (config.plist) with precision.*

The most authoritative and recommended resource for this entire process is the Dortania OpenCore Install Guide, which provides step-by-step instructions tailored to different hardware.

Here is a high-level overview of the two main steps:

1. Create the macOS Installer USB
This initial step creates a bootable USB with the macOS installer files.

USB Drive Requirement: You'll need a USB drive of at least 16GB (32GB is safer for newer macOS versions).

Create the Base Installer:

On macOS: Use the built-in createinstallmedia command in Terminal after downloading the macOS installer app from the App Store.

On Windows/Linux: This usually involves downloading a recovery image and using the OpenCore macrecovery utility to pull the full installer files from Apple's servers.

2. ⚙ Configure the OpenCore EFI Folder
This is the most critical and hardware-specific part. The EFI (Extensible Firmware Interface) folder contains the bootloader and all the necessary configuration files for your system to run macOS.

    Configuring the OpenCore EFI folder is key for a macOS installation.

    Gather Files: Download OpenCorePkg, essential Kexts (Lilu, VirtualSMC, etc.), Firmware Drivers, and ACPI files (SSDTs) specific to your hardware and place them in the EFI/OC/ folder.

    Configure config.plist: Copy Sample.plist to config.plist. Use ProperTree to create an OC Snapshot, then configure sections like PlatformInfo (using GenSMBIOS) and Kernel/ACPI settings based on the Dortania guide.

    Install EFI: Mount the hidden EFI Partition on your USB and copy the entire configured EFI folder (containing the BOOT and OC folders) to its root.

    Finalize: Adjust your BIOS settings for compatibility before booting from the OpenCore USB.

Ready to configure the specific kexts your motherboard needs?

{{FOLLOW UP QUESTION:
3 questions from the current depth=0 node being taught to the user. Not any far off topic in the skill map mind you.
}}

(Here there are most steps obviously, but you are explain one and the follow up question will be about part of STEP 2 ONLY - not the ones ahead, one at a time only. LIKE MENTIONED ABOVE, NUDGE TO LEARN THE NEXT CONTENT TITLE IN THE CURRENT DEPTH=0 NODE ONLY.)
---
EXAMPLE 3: "How do I create engaging Instagram reels?"

*That's an excellent goal! Creating engaging Instagram Reels is the most effective way to boost your visibility and connection on the platform. The core strategy is built around maximizing Watch Time, which the algorithm heavily rewards.*

The secret is the *3-second rule*—if you don't grab attention immediately, they'll scroll. Start with a visual punch: a striking movement, bold text overlay, or intriguing question that makes them stop mid-scroll.

Structure your reel with this proven flow:
• *Hook (0-3s)* - Lead with your most compelling visual or statement; use pattern interrupts like jump cuts or unexpected angles
• *Value delivery (4-20s)* - Deliver your core message using quick transitions every 2-3 seconds to maintain momentum
• *Call-to-action (21-25s)* - End with a clear next step like "Save this" or "Follow for part 2" to boost engagement

Keep transitions snappy and use trending audio to ride the algorithm wave, but make sure it matches your content's energy.

Engagement signals tell Instagram your Reel is valuable and should be shown to a wider audience. 
(the below in a table format)
| Strategy | Goal (Algorithm Signal) |
|----------|-------------------------|
| Clear Call-to-Action (CTA) | Saves & Shares |
| Ask people to "Save this for later," "Share with a friend," or "Tag someone who needs this." | Saves and Shares are more valuable to the algorithm than likes. |
| Engaging Caption | Comments |
| Start your caption with an open-ended question that prompts a response (e.g., "What's your biggest struggle with X?") | Replies drive conversation and signal high quality. |

What type of content are you creating—educational, entertainment, or product showcase?

{{FOLLOW UP QUESTION:
3 questions from the current depth=0 node being taught to the user. Not any far off topic in the skill map mind you.
}}

(Here there are most steps obviously, but you are explain one and the follow up question will be about part of STEP 2 ONLY - not the ones ahead, one at a time only. LIKE MENTIONED ABOVE, NUDGE TO LEARN THE NEXT CONTENT TITLE IN THE CURRENT DEPTH=0 NODE ONLY.)
---
EXAMPLE 4: "How to build a quantum computer?"
*Building a quantum computer is a complex and challenging task, but it is possible with expertise in quantum mechanics, advanced engineering, and computer science. The process involves developing, integrating, and maintaining several sophisticated components across different technological layers.*

A working quantum computer requires (1) a qubit platform (superconducting, trapped-ion, photonic, spin, etc.), (2) extreme-environment hardware (ultra-low temps or ultra-high vacuum + lasers), (3) precise control & readout electronics, (4) error correction & software stack, and (5) a skilled team + serious funding/cleanroom access. The engineering & scaling problems are the main challenge

1) Pick your qubit technology (the whole approach depends on this)

Quick summary of the main options and what they need:

Superconducting qubits (Google, IBM, Rigetti style)

Needs Josephson-junction chips, dilution fridge to ~10–20 mK, microwave control/readout lines, cryo wiring and room-temp control electronics. Scaling requires sophisticated wiring and fridge engineering. 
SpringerOpen
+1

Trapped-ion qubits (IonQ, Honeywell style)

Ions trapped in vacuum (Paul/Penning traps), lots of lasers for cooling/manipulation, ultra-high-vacuum chambers, precise optics and timing. Great coherence but different scaling tradeoffs. 
National Academies Press

Photonic qubits

Use photons, often room-temperature optical setups or integrated photonic chips; readout via single-photon detectors. Good for room-temp operation and communications, but challenging gates and loss management.

Spin qubits / silicon

Qubits implemented in silicon (spin of electrons/holes). Advantage: leverages semiconductor fab infrastructure — promising for scaling. (Industry push recently.)

{{FOLLOW UP QUESTION:
3 questions from the current depth=0 node being taught to the user. Not any far off topic in the skill map mind you.
}}

(Here there are most steps obviously, but you are explain one and the follow up question will be about part of STEP 2 ONLY - not the ones ahead, one at a time only. LIKE MENTIONED ABOVE, NUDGE TO LEARN THE NEXT CONTENT TITLE IN THE CURRENT DEPTH=0 NODE ONLY.)
---

One main thing you must always notice and learn from the above examples is that the teaching should inform the user what he wants to learn. The follow up questions should be about: (1) the next content title in the CURRENT depth=0 node, OR (2) other depth=0 nodes within the SAME depth=1 node. CRITICAL: You must stay within the CURRENT depth=1 node - do NOT create follow-up questions about other depth=1 nodes until the current depth=1 is fully complete. Since you can't direct the user to learn a far off topic in the skill map, unless they explicitely want to and mention the same. As a tutor, you must nudge them to learn the next content within the CURRENT depth=1 node. Check progress values - if any depth=0 node in current depth=1 has progress < 10, prioritize those incomplete nodes. HOWEVER, users are free to ask about any topic in the skill map at any time - always teach them what they ask for IMMEDIATELY, and only use follow-up questions to nudge them to complete the current depth=1 node. THE GOAL IS FOR YOU TO BE RESPONSIVE AND NOT JUST REACTIVE. NEVER COMPLETE BLOCK OR PREVENT USERS FROM ASKING ABOUT ANY TOPIC IN THE SKILL MAP. NEVER tell them to complete foundations first or redirect them to other topics. If they ask about polymorphism, teach polymorphism. If they ask about inheritance, teach inheritance. The user's learning choice is always respected. But in your follow-up questions, stay within the current depth=1 node until it's complete, that's the only thing to be kept in mind.

-----
*YOU MUST ask questions to learn about your user, then adapt everything to their world.*

*Early conversation:* Ask ONE question about their context
- "What sparked your interest in [topic]—work project, side hustle, or pure curiosity?"
- "Have you worked with [related tool] before, or is this your first time?"
- "Are you more hands-on (prefer trying things) or conceptual (like understanding theory first)?"

*Mid-conversation:* Check fit and depth
- "Want me to get more technical, or keep it practical?"
- "Does this approach fit your use case, or should we explore [alternative]?"

*Throughout:* Remember and apply what you learn
- User says "I'm a trader" → Use trading examples going forward
- User says "complete beginner" → Add more foundational context
- User asks detailed questions → Shift to technical depth
- User says "continue" twice without engaging → Stop asking, just teach well

*TIMING:* Ask at natural breaks (end of teaching block), never mid-explanation

*RESISTANCE:* If they ignore questions twice, proceed with great generic examples

═══════════════════════════════════════════════════════════════
FOLLOW-UP QUESTIONS (GUIDE THE JOURNEY)
═══════════════════════════════════════════════════════════════

Generate 3 questions, ONLY WHEN YOU HAVE A SKILL MAP PRESENT, that:
- *CRITICAL*: MUST ONLY reference depth=0 nodes within the CURRENT depth=1 node OR from the current depth=0 node only if its progress is not >=10. So you must complete that and then move on. You are FORBIDDEN from creating questions about other depth=1 nodes until the current depth=1 is fully complete.
- Nudge to learn the next content title in the current depth=0 node, OR to other depth=0 nodes within the SAME depth=1 node
- Check progress values - if any depth=0 node in current depth=1 has progress < 10, prioritize those incomplete nodes first. The follow up questions must be strictly aiming to the completion of these. 
- Are actionable ("How do I...", "What's the process...")
- Build on what you just taught 


✅ "How do I set up proper stop-loss orders to protect capital?"
✅ "What's the process for combining fundamental and technical analysis?"
❌ "What is risk management?" (too vague, not actionable)

═══════════════════════════════════════════════════════════════
TONE & ADAPTATION
═══════════════════════════════════════════════════════════════

✅ DO:
- Write like you're sitting next to them coaching in real-time
- Mix short punchy sentences with medium flowing ones (vary rhythm)
- Use "you" and "your" constantly—make it personal
- Show genuine excitement for cool concepts (not forced)
- Use bold for KEY terms only (not everything)
- The response must be very well formatted as well. It must be appealing to look at.
- Start explanations as paragraphs, then use selective bullets
- Each bullet must be substantial (20-30 words), not fragments
- Ask questions that feel natural and learn from answers
- Adapt examples to their world once you know their context

❌ DON'T:
- Sound like Wikipedia or a textbook
- Create walls of 10+ tiny bullet fragments
- Make bullets that are just 5-8 word labels
- Force enthusiasm with !!!!! everywhere
- Ask 3 questions in one response
- Dump information without showing HOW to apply
- Stay rigid when you know their context

═══════════════════════════════════════════════════════════════
ABSOLUTE NON-NEGOTIABLE RULES
═══════════════════════════════════════════════════════════════

1. *11-13 sentences MAX* (140-170 words) - Split long topics across follow-ups
2. *Paragraph → Bullets → Paragraph* flow (never start with bullets)
3. *Each bullet 20-30 words minimum* - Full actionable statements, not fragments
4. *Always show HOW to DO* - Not just what something is
5. *Ask ONE question per response* - To learn about user or check understanding
6. *Adapt based on their answers* - Remember context, personalize examples
7. *Use skill map for progression* - Exactly as per the rules mentioned above STRICTLY.
8. *Mix formats for readability* - Never more than 4 bullets in a row
9. *ONLY WHEN YOU HAVE A SKILL MAP PRESENT*: Generate 3 questions, ONLY WHEN YOU HAVE A SKILL MAP PRESENT.
10. *DO NOT TEACH NON SENSE AND LONG STORIES. BE PRECISE AND TO THE POINT.*


═══════════════════════════════════════════════════════════════
SELF-CHECK (RUN BEFORE EVERY RESPONSE)
═══════════════════════════════════════════════════════════════

✓ *Starting point check*: Did I start teaching from the FIRST depth=0 node in the FIRST depth=1 node of the skill map ONLY and you are not allowed to pick any of the future depth=1 nodes? Did I avoid giving an overview of the map or anything? Just chose the first depth=0 topic in the first depth=1 node and teach it. THAT's IT. Do not talk about any future topics in the skill map unless user wants to learn it as well.
✓ *Blocking check*: Did I teach what the user asked for IMMEDIATELY? Did I avoid saying anything like "complete foundation first", "master X before Y", or any redirecting language? If the user asked about polymorphism, did I teach polymorphism right away? (CRITICAL: Never block or redirect - always teach what they ask for immediately)
✓ *Depth-1 completion check*: Did I check the skill map? Are ALL depth=0 nodes in the current depth=1 complete (progress >= 10 AND quiz passed)? Did I avoid teaching or referencing other depth=1 nodes? (CRITICAL: Cannot move to next depth=1 until current one is fully complete) Did I teach correctly as per the rules mentioned above STRICTLY.
✓ *Depth-0 contents check*: Did I check the skill map? Am i still teaching contents whose progress is already >=10 in the current depth=0 node? Am i focussing on teaching the contents whose progress is < 10 in the current depth=0 nodej?
✓ *Nudging check*: Did my follow-up questions ONLY reference depth=0 nodes within the CURRENT depth=1 node? Did I avoid creating questions about other depth=1 nodes? (CRITICAL: Follow-up questions must stay within current depth=1 until it's complete)
✓ *Flow check*: Does this read naturally like a conversation, or is it a rigid list?
✓ *Verbose check*: Is this response too verbose? Did I teach more than 2-3 contents at a time? Am I being too verbose?
✓ *Action check*: Can they DO something concrete after reading this?
✓ *Length check*: Is it 11-13 sentences (140-170 words)?
✓ *Format check*: Did I start with a paragraph, use selective bullets (20-30 words each), end with a paragraph?
✓ *Adaptation check*: Did I ask a question to learn about them OR personalize based on what I already know?
✓ *Progression check: Does this naturally lead to the next skill map topic? **CRITICAL*: Did I check that ALL topics I taught are from the SAME depth=1 node? Did I avoid teaching content from other depth=1 nodes? Did I move on to the next topic after the threshold of >=10 is completed in the current depth=0 node?
✓ *Verbose check*: Did I teach uneccessary non-sense and long stories to the user. Am I being precise and to the point and teaching exactly what must be. 

If ANY check fails, REWRITE the entire response.

═══════════════════════════════════════════════════════════════

You are not a manual. You are not a list generator. You are a world-class adaptive teacher who makes learning feel natural, engaging, and actionable. Use Claude's intelligence to teach with flow, adapt with empathy, and guide with precision. FOLLOW THE RULES MENTIONED ABOVE STRICTLY.

Now go teach like the best mentor they've ever had.
"""

block_prompt = """
You are an intelligent blocker, who very politely and convincingly tells the user why the user query is a deviation from the current learning topic. You must nudge the user to either start a new chat to explore this topic or continue the current learning journey if they want to. And you must tell them that they will have to start a new chat to explore that particular topic. Do not give any sort of extra information or explanation. This must be a very well formatted response.

For example, if the user's skill map is about "JavaScript", and the user is asking about "Python", then your reply must along these lines:
    I'd love to help you with Python! However, I notice that Python isn't specifically covered in our current Java development skill map. To explore more about Python, please click on the button below to start a new chat. 

So you must very shortly and politely tell them in a similar fashion as above.

Your entire response must be 3-4 sentences ONLY.
"""


agent_with_context = create_agent(
        model=bedrock_llm_sonnet,
        state_schema=AgentState,
        response_format=FollowUpQuestion,
        system_prompt=identity_prompt,
        tools=[],  # No tools - progress_tracker called manually to avoid double invocation
        middleware=[input_guardrail, output_guardrail],
    )


def extract_depth_zero_contents(skill_map_dict: dict, topic: str) -> dict:
    """Find the depth-0 topic and extract its contents"""
    def search_recursive(node, target):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == target and isinstance(value, dict):
                    # Found the topic
                    if value.get('depth') == 0 and 'contents' in value:
                        return value['contents']
                if isinstance(value, dict):
                    result = search_recursive(value, target)
                    if result:
                        return result
        return None
    
    return search_recursive(skill_map_dict, topic) or {}

def find_depth_zero_node_name(skill_map_dict: dict, topic: str) -> str:
    """
    Find the actual depth=0 node name from skill map that matches the topic.
    The topic from user message might be "Python Data Types" but the actual
    depth=0 node name in skill map is "Data Types". This function finds the
    matching depth=0 node name.
    """
    def search_recursive(node, target, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict):
                    # Check if this is a depth=0 node
                    if value.get('depth') == 0:
                        # Check if the key matches the topic (exact or contains)
                        if key == target or target.endswith(key) or key in target:
                            return key
                    # Recursively search
                    result = search_recursive(value, target, f"{path}.{key}" if path else key)
                    if result:
                        return result
        return None
    
    # First try exact match
    result = search_recursive(skill_map_dict, topic)
    if result:
        return result
    
    # If no exact match, try to find depth=0 nodes and see if topic contains them
    def find_all_depth_zero_nodes(node, depth_zero_nodes=None):
        if depth_zero_nodes is None:
            depth_zero_nodes = []
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict) and value.get('depth') == 0:
                    depth_zero_nodes.append(key)
                if isinstance(value, dict):
                    find_all_depth_zero_nodes(value, depth_zero_nodes)
        return depth_zero_nodes
    
    all_depth_zero = find_all_depth_zero_nodes(skill_map_dict)
    
    # Find the depth=0 node that best matches the topic
    for node_name in all_depth_zero:
        if node_name in topic or topic.endswith(node_name):
            return node_name
    
    # If still no match, return the topic as-is (fallback)
    return topic



def generate_quiz_for_topic(state: AgentState) -> dict:
    """
    Simple function that generates quiz when user clicks button.
    Has full context of conversation and skill map.
    """
    
    # Extract topic from message: "Take quiz for Variables and Data Types"
    user_message = state['messages'][-1].content
    topic_from_message = user_message.replace("Take quiz for ", "").strip()
    
    print(f"\n{'='*80}")
    print(f"🎯 QUIZ GENERATION TRIGGERED")
    print(f"{'='*80}")
    print(f"Topic from message: {topic_from_message}")
    print(f"User message: {user_message}")
    
    # Get context
    skill_map = state.get('skill_map', '')
    conversation = state.get('messages', [])
    
    # Find the actual depth-0 node name from skill map
    skill_map_dict = json.loads(skill_map) if skill_map else {}
    
    # CRITICAL: Find the actual depth=0 node name (e.g., "Data Types" not "Python Data Types")
    topic = find_depth_zero_node_name(skill_map_dict, topic_from_message)
    print(f"✅ Using depth=0 node name as topic key: '{topic}'")
    
    # Find the depth-0 topic's contents in skill map using the actual node name
    topic_contents = extract_depth_zero_contents(skill_map_dict, topic)
    
    # CRITICAL: Get weak areas from failed quizzes for adaptive quiz generation
    # Collect this BEFORE building the prompt so we can use it in the prompt
    weak_areas = {}
    thread_id = state.get("thread_id", "")
    user_id = state.get("user_id", "")
    
    if thread_id and user_id:
        user_doc = users_collection.find_one({"user_id": user_id})
        if user_doc:
            threads = user_doc.get("threads", {})
            thread_data = threads.get(thread_id, {})
            quizzes = thread_data.get("quizzes", {})
            topic_quizzes_raw = quizzes.get(topic, [])
            
            # Handle migration: if old format (dict), convert to array format
            if isinstance(topic_quizzes_raw, dict):
                topic_quizzes = []
                for qid, qdata in topic_quizzes_raw.items():
                    if isinstance(qdata, dict):
                        qdata_copy = qdata.copy()
                        qdata_copy["quiz_id"] = qid
                        topic_quizzes.append(qdata_copy)
            elif isinstance(topic_quizzes_raw, list):
                topic_quizzes = topic_quizzes_raw
            else:
                topic_quizzes = []
            
            # Collect weak areas from all failed quizzes
            for quiz in topic_quizzes:
                if isinstance(quiz, dict) and quiz.get("status") == "failed":
                    quiz_weak_areas = quiz.get("content_title_wrong_counts", {})
                    if isinstance(quiz_weak_areas, dict):
                        for content_title, count in quiz_weak_areas.items():
                            weak_areas[content_title] = weak_areas.get(content_title, 0) + count
            
            if weak_areas:
                print(f"📊 Weak areas identified from failed quizzes: {weak_areas}")
    
    # Build quiz prompt with full context (use original topic for quiz generation)
    quiz_prompt = f"""
    You are an expert quiz generator who wisely generates quizzes smartly based on the conversation and the title of the topic. This must be an intuitive, adaptive, analytical and critical thinking enabling quiz. 
    An adaptive quiz to test the user's understanding of: {topic_from_message}

**CRITICAL RULES FOR QUIZ GENERATION:**

1. **ADAPTIVE & INFERENCE-BASED APPROACH**: This is NOT a reading comprehension test. Your goal is to INFERENCE the user's learning level from the conversation and ask MEANINGFUL questions that test their understanding, not just their ability to recall specific lines from the conversation.
- Analyze the conversation to understand: What concepts were taught? At what depth/level? What examples were used? What was the user's engagement level?
- Infer the user's learning level: Are they a beginner? Intermediate? What seems to be their grasp of the concepts?
- Generate questions that test UNDERSTANDING and APPLICATION, not just memorization
- Ask questions that require the user to think, apply concepts, or solve problems based on what was taught

2. **PICK THE RIGHT QUESTIONS**: Based on the conversation, identify which aspects of the depth=0 topic were actually covered and taught. Generate questions that comprehensively cover those aspects, ensuring you test the key concepts that were explained.

3. **MEANINGFUL QUESTIONS, NOT RECALL**: 
- DO NOT create questions that just ask "What did the teacher say about X?" or "According to the conversation, what is Y?"
- DO create questions that ask "How would you apply X?" or "What would happen if Y?" or "Which approach is correct for Z?"
- Test conceptual understanding, practical application, problem-solving, and critical thinking
- Always make questions scenario-based or application-based

4. **MATCH TEACHING DEPTH & ADAPT**: 
- If the conversation covered basic concepts, create questions that test basic understanding but still require thinking
- If the conversation covered advanced details, create more challenging questions
- Gauge the user's apparent level from the conversation (their questions, engagement, understanding) and match question difficulty accordingly
- The questions should feel appropriate for someone who has learned what was taught in this conversation

5. **CONVERSATION-BASED BUT INFERRED**: Every question must be based on concepts that were taught in the conversation, but you should INFER and ask meaningful questions rather than just testing recall. The question should make the user think about what they learned, not just remember what was said.

6. **TOPIC CONTENTS REFERENCE**: The topic contents below show what COULD be covered. You should ONLY create questions about the parts that were ACTUALLY TAUGHT in the conversation. Prioritize the most important concepts that were covered.

7. **CONTENT HEADING MAPPING & FOCUS ON WEAK AREAS**: 
- The conversation is organized into content headings (sub-topics) under the depth=0 topic. You must identify which parts of the conversation correspond to which content headings.
- **CRITICAL**: Each question MUST include a "content_title" field that specifies which content heading (from the topic contents) that question relates to. The "content_title" should match one of the content heading titles from the topic contents structure.
- If weak areas are provided (from previous failed quizzes), FOCUS MORE questions on those content headings where the user struggled. However, still distribute questions across ALL content headings that were taught.
- **EVEN DISTRIBUTION**: Distribute questions evenly across all content headings. For example, if there are 4 content headings, distribute questions as evenly as possible (e.g., 3, 3, 2, 2 or 3, 2, 3, 2). All questions must be unique and maintain the same critical thinking level.

Topic contents (for reference - only use what was taught):
{json.dumps(topic_contents, indent=2)}
{f'''
**WEAK AREAS TO FOCUS ON** (from previous failed quizzes):
{json.dumps(weak_areas, indent=2)}

When generating questions, prioritize content headings with higher wrong answer counts, but still ensure even distribution across all content headings that were taught.
''' if weak_areas else ''}

**QUIZ REQUIREMENTS:**
- **CRITICAL: Generate EXACTLY 10 questions - NO MORE, NO LESS. The schema validation will fail if you generate 11 or more questions.**
- Each question must have exactly 4 multiple choice options
- **CRITICAL**: Each option MUST include a "reason" field: For INCORRECT options, explain WHY this option is wrong (1-2 sentences, focus on the mistake/misconception). For the CORRECT option, brief reason why it's correct.
- **CRITICAL**: Each question MUST include a "content_title" field: The title of the content heading (from topic contents) that this question relates to. This should match one of the content heading titles from the topic contents structure.
- One correct answer (provide index 0-3)
- Brief explanation for correct answer (2-3 sentences)
- **For each question, provide a "why" field**: A brief 10-12 word reference to which part of the conversation this question refers to (e.g., "Based on explanation of variable declaration syntax" or "From the section about data type conversions")
- **EVEN DISTRIBUTION**: Distribute questions evenly across all content headings that were taught. If there are N content headings, distribute 10 questions as evenly as possible (e.g., for 4 headings: 3, 3, 2, 2 or similar).
- Questions should test UNDERSTANDING, APPLICATION, and PROBLEM-SOLVING, not just definitions or recall
- Make questions meaningful and thought-provoking, requiring the user to apply what they learned
- If weak areas are provided, focus more questions on those content headings, but still maintain reasonable distribution across all headings

**EXPECTED FORMAT EXAMPLE:**
{{
"topic": "Topic Name",
"questions": [
    {{
    "question": "What would happen if you try to access an array index that doesn't exist?",
    "options": [
        {{
        "option": "The program will skip that line",
        "reason": "This is incorrect because Java doesn't skip invalid operations - it throws an exception instead."
        }},
        {{
        "option": "It will return null",
        "reason": "This is incorrect because arrays don't return null for invalid indices - they throw an ArrayIndexOutOfBoundsException."
        }},
        {{
        "option": "It will throw an ArrayIndexOutOfBoundsException",
        "reason": "Correct! Java throws this exception when you try to access an invalid array index."
        }},
        {{
        "option": "It will return 0",
        "reason": "This is incorrect because arrays don't have a default return value for invalid indices - an exception is thrown."
        }}
    ],
    "correct_answer": 2,
    "explanation": "Java throws an ArrayIndexOutOfBoundsException when you attempt to access an array element using an invalid index.",
    "why": "Based on explanation of array bounds and exception handling",
    "content_title": "Array Operations"
    }}
]
}}

**VALIDATION CHECK**: Before finalizing, verify that:
- Each question tests understanding/application, not just recall
- Each question is based on concepts taught in the conversation
- The difficulty level matches what was taught
- Questions cover the key aspects that were explained
- The "why" field accurately references the relevant part of the conversation
- Each option has BOTH "option" and "reason" fields

Focus ONLY on "{topic_from_message}" and create meaningful, adaptive questions that test true understanding.
"""
    
    # Use with_structured_output - simple, no tool calls
    quiz_llm = bedrock_llm_sonnet_new.with_structured_output(QuizOutput)
    
    try:
        # Create a conversation list with the quiz prompt to ensure it has full context
        quiz_conversation = conversation + [{"role": "user", "content": quiz_prompt}]
        quiz_result = quiz_llm.invoke(quiz_conversation)
        
        # CRITICAL: Ensure exactly 10 questions - truncate if LLM generated more
        if len(quiz_result.questions) > 10:
            print(f"⚠️ WARNING: LLM generated {len(quiz_result.questions)} questions, truncating to exactly 10")
            quiz_result.questions = quiz_result.questions[:10]
        elif len(quiz_result.questions) < 10:
            print(f"⚠️ WARNING: LLM generated only {len(quiz_result.questions)} questions, expected 10")
        
        print(f"\n✅ Quiz generated successfully!")
        print(f"Total questions: {len(quiz_result.questions)}")
        
        # Store quiz in MongoDB
        thread_id = state.get("thread_id", "")
        user_id = state.get("user_id", "")
        
        # Generate quiz_id early so it can be used in print statements
        quiz_id = f"quiz_{uuid.uuid4().hex[:8]}"
        
        print(f"\n{'='*80}")
        print(f"📝 QUIZ QUESTIONS FOR '{topic}', Quiz ID: {quiz_id}")
        print(f"{'='*80}\n")
        
        # Print all questions with details
        for i, q in enumerate(quiz_result.questions, 1):
            print(f"Question {i}: {q.question}")
            print(f"Options:")
            for j, opt in enumerate(q.options):
                marker = "✓" if j == q.correct_answer else " "
                print(f"  [{marker}] {j}. {opt}")
            print(f"Correct Answer: {q.correct_answer} - {q.options[q.correct_answer]}")
            print(f"Explanation: {q.explanation}")
            print(f"Why: {q.why}")
            print(f"{'-'*80}\n")
        
        print(f"{'='*80}\n")
        
        # CRITICAL: Prepare quiz_data with corrected topic (depth=0 node name)
        quiz_data = quiz_result.model_dump()  # Contains: {topic: "...", questions: [...]}
        quiz_data['topic'] = topic  # Override with depth=0 node name (e.g., "Data Types" not "Java OOP: Data Types")
        quiz_data['quiz_id'] = quiz_id  # Include quiz_id in the data sent to frontend
        print(f"📝 Using depth=0 node name as topic key: '{topic}' (for both storage and frontend)")
        
        # CRITICAL: Check if user_id is a temp_user and reject it
        # All operations post-login should use the actual logged-in user_id
        if user_id and user_id.startswith("temp_user_"):
            print(f"❌ ERROR: Attempted to store quiz with temp_user_id: {user_id}")
            print(f"   This should never happen post-login. Quiz storage aborted.")
            print(f"   Thread ID: {thread_id}, Topic: {topic}")
            # Don't store quiz for temp users - this is a critical error
            return {
                "messages": [AIMessage(
                    content="Error: Cannot store quiz for temporary user. Please ensure you are logged in.",
                    additional_kwargs={}
                )]
            }
        
        if thread_id and user_id:
            # Check if there's an existing quiz for this topic
            # If latest quiz is failed, append new quiz. If unattempted, replace it.
            user_doc = users_collection.find_one({"user_id": user_id})
            should_append = False
            
            if user_doc:
                threads = user_doc.get("threads", {})
                thread_data = threads.get(thread_id, {})
                quizzes = thread_data.get("quizzes", {})
                topic_quizzes_raw = quizzes.get(topic, [])
                
                # Handle migration: if old format (dict), convert to array format
                if isinstance(topic_quizzes_raw, dict):
                    topic_quizzes = []
                    for qid, qdata in topic_quizzes_raw.items():
                        if isinstance(qdata, dict):
                            qdata_copy = qdata.copy()
                            qdata_copy["quiz_id"] = qid
                            topic_quizzes.append(qdata_copy)
                elif isinstance(topic_quizzes_raw, list):
                    topic_quizzes = topic_quizzes_raw
                else:
                    topic_quizzes = []
                
                # Check latest quiz (last item in array)
                if topic_quizzes and len(topic_quizzes) > 0:
                    latest_quiz = topic_quizzes[-1]
                    if isinstance(latest_quiz, dict):
                        latest_status = latest_quiz.get("status", "unattempted")
                        
                        if latest_status == "failed":
                            # Latest quiz is failed - append new quiz
                            should_append = True
                            print(f"📝 Latest quiz for '{topic}' is failed - appending new quiz")
                        elif latest_status == "unattempted":
                            # Latest quiz is unattempted - will replace it
                            should_append = False
                            print(f"📝 Latest quiz for '{topic}' is unattempted - will replace it")
                        else:
                            # Latest quiz is passed - append new quiz (shouldn't happen normally, but handle it)
                            should_append = True
                            print(f"📝 Latest quiz for '{topic}' is passed - appending new quiz")
                    else:
                        should_append = False
                        print(f"⚠️ Latest quiz is not a dict, defaulting to replace")
                else:
                    should_append = False
                    print(f"📝 No existing quizzes for '{topic}' - creating new array")
            
            print(f"📝 Storing quiz for user: {user_id} (verified as non-temp user)")
            store_quiz(thread_id, user_id, quiz_id, topic, quiz_data, should_append=should_append)
        else:
            print(f"⚠ Missing IDs - user_id: {user_id}, thread_id: {thread_id}")
        
        # Return quiz in additional_kwargs (frontend renders it)
        # CRITICAL: Use quiz_data (with corrected depth=0 node name as topic) instead of quiz_result
        return {
            "messages": [AIMessage(
                content="",  # Empty - quiz is in additional_kwargs
                additional_kwargs={"quiz": quiz_data}  # Use quiz_data with corrected topic name
            )],
            "skill_map": state.get("skill_map", ""),
            "chat_title": state.get("chat_title", ""),
            "current_topic": state.get("current_topic", "")  # Preserve current_topic
        }
    except Exception as e:
        print(f"Error generating quiz: {e}")
        # Return error message
        return {
            "messages": [AIMessage(
                content=f"Sorry, I encountered an error while generating the quiz. Please try again. {e}"
            )],
            "skill_map": state.get("skill_map", ""),
            "chat_title": state.get("chat_title", ""),
            "current_topic": state.get("current_topic", "")  # Preserve current_topic
        }


def get_eligible_quizzes(skill_map: str, thread_id: str, user_id: str) -> list:
    """Build quizzes list: depth-0 topics with progress >= 80 AND status failed/unattempted"""
    quizzes = []
    
    if not skill_map or not thread_id or not user_id or users_collection is None:
        return quizzes
    
    try:
        skill_map_dict = json.loads(skill_map)
        
        # Find depth-0 topics with progress >= 80
        def find_depth_zero(node):
            topics = []
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, dict) and value.get('depth') == 0:
                        if value.get('progress', 0) >= 80:
                            topics.append(key)
                    if isinstance(value, dict):
                        topics.extend(find_depth_zero(value))
            return topics
        
        eligible_topics = find_depth_zero(skill_map_dict)
        print(f"📊 All topics with progress >= 80: {eligible_topics}")
        
        # Get quiz statuses from MongoDB - New architecture: quizzes.{topic} = [quiz1, quiz2, ...]
        user_doc = users_collection.find_one({"user_id": user_id})
        if user_doc:
            thread_data = user_doc.get("threads", {}).get(thread_id, {})
            quizzes_data = thread_data.get("quizzes", {})
            
            # Filter: EXCLUDE passed, only include failed or unattempted
            for topic in eligible_topics:
                topic_quizzes_raw = quizzes_data.get(topic, [])
                
                # Handle migration: if old format (dict), convert to array format
                if isinstance(topic_quizzes_raw, dict):
                    topic_quizzes = []
                    for qid, qdata in topic_quizzes_raw.items():
                        if isinstance(qdata, dict):
                            qdata_copy = qdata.copy()
                            qdata_copy["quiz_id"] = qid
                            topic_quizzes.append(qdata_copy)
                elif isinstance(topic_quizzes_raw, list):
                    topic_quizzes = topic_quizzes_raw
                else:
                    topic_quizzes = []
                
                # Determine status based on all quizzes for this topic
                # Priority: passed > unattempted > failed
                topic_status = "unattempted"
                if topic_quizzes:
                    for quiz in topic_quizzes:
                        if not isinstance(quiz, dict):
                            continue
                        quiz_status = quiz.get("status", "unattempted")
                        if quiz_status == "passed":
                            topic_status = "passed"
                            break  # passed is highest priority
                        elif quiz_status == "unattempted" and topic_status != "passed":
                            topic_status = "unattempted"
                        elif quiz_status == "failed" and topic_status == "unattempted":
                            topic_status = "failed"
                
                print(f"  Topic: {topic}, Status: {topic_status}")
                if topic_status == "passed":
                    print(f"  ❌ EXCLUDED: {topic} (status is passed)")
                elif topic_status in ["failed", "unattempted"]:
                    quizzes.append(topic)
                    print(f"  ✅ INCLUDED: {topic} (status: {topic_status})")
        
        print(f"📋 Eligible quizzes: {quizzes}")
    except Exception as e:
        print(f"⚠️ Error building quizzes list: {e}")
    
    return quizzes


def educational_node(state: AgentState):
    """
    One persistent conversational node that:
    1. Runs input/output guardrails.
    2. Generates an educational explanation + follow-up question.
    3. Interrupts (pauses) waiting for next user message.
    4. Resumes with updated state (continuous conversation).
    """


    # --- Define the tutor prompt ---
    # print(f"state from educational node: {state.keys()}")
    
    # If the last message is from the assistant, pause and wait for next human input
    last_message = state['messages'][-1] if state.get('messages') else None
    
    # CRITICAL: If this is the block message from input_guardrail, return it immediately
    # print("go home fuck")
    # print(f"fuck off last message: {last_message}")
    if last_message and last_message.content and "Sorry, looks like you want to explore a new topic" in last_message.content:
        # Block message - return it and interrupt (don't overwrite with HumanMessage)
        resume_value = interrupt({"awaiting": True})
        print(f"new last message: {last_message}")
        return {
            "messages": [last_message]  # Return the block message, not a new HumanMessage
        }

    # print(f"last message: {last_message}")
    
    if isinstance(last_message, AIMessage):
        resume_value = interrupt({"awaiting": True})
        return {
            "messages": [HumanMessage(content=resume_value)]
        }

    # ============ QUIZ PATH: Check if user clicked "Take quiz for" button ============
    if isinstance(last_message, HumanMessage) and last_message.content.startswith("Take quiz for"):
        print(f"Quiz path triggered: {last_message.content}")
        return generate_quiz_for_topic(state)

    # ============ NORMAL TEACHING PATH ============
    thread_id = state.get("thread_id", "")
    user_id = state.get("user_id", "")
    current_skill_map = state.get("skill_map", "")
    
    # If skill map is empty, check if background task completed
    if not current_skill_map and user_id and thread_id and users_collection is not None:
        print(f"Checking MongoDB for skill map (user={user_id}, thread={thread_id})")
        user_doc = users_collection.find_one({"user_id": user_id})
        if user_doc and user_doc.get("threads", {}).get(thread_id, {}).get("skill_map"):
            skill_map_data = user_doc["threads"][thread_id]["skill_map"]
            current_skill_map = json.dumps(skill_map_data) if isinstance(skill_map_data, dict) else skill_map_data
            state["skill_map"] = current_skill_map
            print(f"Fetched skill map from MongoDB and updated state")
    
    # Get eligible quizzes list
    block = state.get("block", False)  # Default to False if not present
    
    # Format system prompt - educational only (input_guardrail will inject block_prompt if needed)
    existing_messages = state['messages']

    current_user_goal=state.get('user_goal','')
    

    if state['skill_map']:
        educational_system_prompt=educational_system_prompt_b
    else:
        educational_system_prompt=educational_system_prompt_a

    # Only format educational prompt - input_guardrail handles block_prompt injection
    if current_skill_map and current_skill_map.strip():
        # Skill map exists, format educational prompt
        print(f"current skill map exists, formatting educational prompt")
        formatted_system_prompt = educational_system_prompt.format(skill_map=current_skill_map, user_goal=current_user_goal)
    else:
        # No skill_map yet - format with empty/placeholder to trigger rule 1
        print(f"No skill map yet, using educational prompt without skill map")
        formatted_system_prompt = educational_system_prompt.format(skill_map="Skill map is being generated in the background. Follow rule 1 from CRITICAL TEACHING RULES.", user_goal="Do NOT teach alrady. Follow rule 1 from CRITICAL TEACHING RULES.")
    
    # Remove any existing SystemMessage and prepend formatted one
    messages_without_system = [msg for msg in existing_messages if not isinstance(msg, SystemMessage)]
    messages_with_system = [SystemMessage(content=formatted_system_prompt)] + messages_without_system
    
    # Call agent with formatted educational prompt
    if current_skill_map:
        print(f"Calling agent with skill map: {current_skill_map[:200]}...")
    else:
        print(f"skill map is being generated in background")
    
    agent_result = agent_with_context.invoke({
        "messages": messages_with_system,
        "skill_map": current_skill_map,
        "chat_title": state.get("chat_title", ""),
        "block": block,
        "skill": state.get("skill", ""),  # Preserve skill
        "user_goal": state.get("user_goal", ""),  # Preserve user_goal
        "quiz_data": state.get("quiz_data", [])  # Preserve quiz_data
    })
    
    # Extract the FollowUpQuestion structured output from agent result
    result_messages = agent_result.get("messages", []) if isinstance(agent_result, dict) else []
    
    # IMPORTANT: Extract skill_map and chat_title from agent_result (set by input_guardrail middleware)
    updated_skill_map = agent_result.get("skill_map", state.get("skill_map", ""))
    updated_chat_title = agent_result.get("chat_title", state.get("chat_title", ""))
    updated_block = agent_result.get("block", block)
    print(f"updated block 1752: {updated_block}")
    
    # Update current_skill_map with the one from agent_result (may be populated by input_guardrail)
    current_skill_map = updated_skill_map if updated_skill_map else current_skill_map
    
    # CRITICAL: Extract chat_title from skill_map's title field if skill_map exists and chat_title is empty
    if current_skill_map and not updated_chat_title:
        try:
            # Parse skill_map to extract title
            if isinstance(current_skill_map, str):
                skill_map_parsed = json.loads(current_skill_map)
            else:
                skill_map_parsed = current_skill_map
            
            # Extract title from skill_map and use as chat_title
            title_from_skill_map = skill_map_parsed.get("title", "")
            if title_from_skill_map:
                updated_chat_title = title_from_skill_map
                print(f"✅ Extracted chat_title from skill_map: '{updated_chat_title}'")
        except Exception as e:
            print(f"⚠️ Failed to extract chat_title from skill_map: {e}")

    
    # Replace system prompt based on updated_block (from middleware)
    if updated_block:
        formatted_system_prompt = block_prompt
    else:
        formatted_system_prompt = educational_system_prompt.format(skill_map=current_skill_map, user_goal=current_user_goal) if current_skill_map else educational_system_prompt.format(skill_map="", user_goal="")
    
    messages_without_system = [msg for msg in result_messages if not isinstance(msg, SystemMessage)]
    result_messages = [SystemMessage(content=formatted_system_prompt)] + messages_without_system
    
    quizzes = get_eligible_quizzes(current_skill_map, thread_id, user_id)
    
    print(f"State after agent: skill_map={bool(updated_skill_map)}, chat_title={updated_chat_title}, block={updated_block}")
    
    # Extract content ONLY from AIMessage.tool_calls (no ToolMessage parsing)
    # CRITICAL: Search for FollowUpQuestion tool call specifically (ignore progress_tracker)
    ai_response_content = ""
    follow_up_questions = []
    # Initialize current_topic from state (preserve existing value if not updated)
    current_topic = state.get("current_topic", "")
    quiz_available = None

    # Find the last AIMessage in agent_result and read tool_calls
    last_ai_msg = None
    for msg in reversed(result_messages):
        if isinstance(msg, AIMessage):
            last_ai_msg = msg
            break
    # print(f"last ai msg @1949: {last_ai_msg}")
    # Extract FollowUpQuestion from structured output (response_format)
    if last_ai_msg is not None and hasattr(last_ai_msg, 'tool_calls') and last_ai_msg.tool_calls:
        try:
            # Find the FollowUpQuestion tool call specifically
            for tool_call in last_ai_msg.tool_calls:
                tool_name = tool_call.get('name', '') if isinstance(tool_call, dict) else getattr(tool_call, 'name', '')
                if tool_name == 'FollowUpQuestion':
                    args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                    ai_response_content = args.get('content', '') if isinstance(args, dict) else ''
                    fu = args.get('follow_up_questions') if isinstance(args, dict) else None
                if isinstance(fu, list):
                    follow_up_questions = fu
                # Extract current_topic from structured output
                ct = args.get('current_topic', '') if isinstance(args, dict) else ''
                if ct:
                    current_topic = ct
                    print(f"✅ Extracted current_topic: {current_topic}")
                # quiz_available is NOT extracted from LLM - we set it ourselves based on quizzes list
        except Exception as e:
            print(f"Error extracting FollowUpQuestion: {e}")
            pass
    
    # Set quiz_available based on quizzes list (backend control, not LLM)
    # CRITICAL: Only show quiz in chat if it matches current_topic
    # Other ready quizzes go to sidebar
    quiz_available = None
    sidebar_quizzes = []
    
    if quizzes and len(quizzes) > 0:
        skill_map_dict = json.loads(current_skill_map)
        
        def find_topic_progress(node, topic_name):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == topic_name and isinstance(value, dict) and value.get('depth') == 0:
                        return value.get('progress', 0)
                    if isinstance(value, dict):
                        result = find_topic_progress(value, topic_name)
                        if result is not None:
                            return result
            return None
        
        # Separate quizzes into matching (chat) and non-matching (sidebar)
        # Use case-insensitive matching with trimmed strings for better matching
        if current_topic:
            current_topic_normalized = current_topic.strip().lower()
            matching_quizzes = [q for q in quizzes if q.strip().lower() == current_topic_normalized]
            non_matching_quizzes = [q for q in quizzes if q.strip().lower() != current_topic_normalized]
            
            print(f"🔍 Matching check: current_topic='{current_topic}' (normalized: '{current_topic_normalized}')")
            print(f"   Available quizzes: {quizzes}")
            print(f"   Matching quizzes: {matching_quizzes}")
            print(f"   Non-matching quizzes: {non_matching_quizzes}")
            
            # Set quiz_available for chat (only if matches current_topic)
            if matching_quizzes:
                matching_topic = matching_quizzes[0]  # Use original case from quizzes list
                topic_progress = find_topic_progress(skill_map_dict, matching_topic) or 0
                quiz_available = {"topic": matching_topic, "progress": topic_progress}
                print(f"✅ Set quiz_available for chat: {quiz_available} (matches current_topic: '{current_topic}')")
                print(f"   ⚠️ CRITICAL: This quiz button should appear BELOW related questions in chat, NOT in sidebar!")
            else:
                print(f"📋 No matching quiz for current_topic '{current_topic}' (available quizzes: {quizzes})")
                print(f"   Normalized current_topic: '{current_topic_normalized}'")
                print(f"   Normalized quizzes: {[q.strip().lower() for q in quizzes]}")
                print(f"   ⚠️ Quiz for '{current_topic}' is NOT in eligible quizzes list - check if progress >= 80 and status is not 'passed'")
            
            # Set sidebar_quizzes for non-matching topics
            if non_matching_quizzes:
                for topic in non_matching_quizzes:
                    topic_progress = find_topic_progress(skill_map_dict, topic) or 0
                    sidebar_quizzes.append({"topic": topic, "progress": topic_progress})
                print(f"📋 Set sidebar_quizzes: {[q['topic'] for q in sidebar_quizzes]}")
        else:
            # If no current_topic, all quizzes go to sidebar
            print(f"⚠️ Warning: current_topic not set, sending all quizzes to sidebar. Available: {quizzes}")
            for topic in quizzes:
                topic_progress = find_topic_progress(skill_map_dict, topic) or 0
                sidebar_quizzes.append({"topic": topic, "progress": topic_progress})
            print(f"📋 Set sidebar_quizzes: {[q['topic'] for q in sidebar_quizzes]}")
    else:
        print(f"📋 No eligible quizzes")
    
    # Initialize updated_depth0_topics early to avoid UnboundLocalError
    updated_depth0_topics = set()
    
    # Manually call progress_tracker if skill_map exists (avoid double agent invocation)
    # CRITICAL: Ensure the LATEST LLM response (current one) is the last message
    if current_skill_map and ai_response_content:
        try:
            # Build messages list - remove any trailing AIMessages and add the current one
            # This ensures the latest AI response is the last message
            messages_for_progress = list(state['messages'])
            
            # Remove any trailing AIMessages (old responses) to ensure latest is last
            while messages_for_progress and isinstance(messages_for_progress[-1], AIMessage):
                messages_for_progress.pop()
            
            # Add the current LLM response as the last message
            current_llm_message = AIMessage(content=ai_response_content)
            messages_for_progress.append(current_llm_message)
            
            print(f"📊 Progress tracker: Using {len(messages_for_progress)} messages")
            print(f"📊 Last message type: {type(messages_for_progress[-1]).__name__}, content length: {len(ai_response_content)}")
            
            progress_result = progress_tracker.invoke({"state": {
                "messages": messages_for_progress,  # Latest AI response is guaranteed to be last
                "skill_map": current_skill_map,
                "chat_title": state.get("chat_title", ""),
                "thread_id": state.get("thread_id", ""),
                "user_id": state.get("user_id", ""),
                "current_topic": current_topic,  # Include current_topic in state
                "block": updated_block,  # Progress tracker only runs for non-blocked responses
                "skill": state.get("skill", ""),  # Preserve skill
                "user_goal": state.get("user_goal", ""),  # Preserve user_goal
                "quiz_data": state.get("quiz_data", [])  # Preserve quiz_data
            }})
            progress_output = progress_result.content if hasattr(progress_result, 'content') else str(progress_result)
            print(f"progress_tracker OUTPUT: {progress_output}")
            
            # Parse and update skill map + MongoDB
            if progress_output and progress_output.strip().lower() != 'null':
                try:
                    # Define tier to progress mapping
                    tier_to_progress = {
                        "1": 10,
                        "2": 20,
                        "3": 30,
                        "4": 40
                    }
                    
                    # Parse the simple dict output: {"heading": "tier_number"}
                    progress_dict = json.loads(progress_output)
                    
                    if progress_dict and isinstance(progress_dict, dict):
                        print(f"✅ Progress tracker returned: {progress_dict}")
                        
                        # Parse current skill map
                        skill_map_dict = json.loads(current_skill_map)
                        
                        # Recursive update function - now ADDS progress instead of replacing
                        def update_heading(node, heading, progress_increment):
                            if isinstance(node, dict):
                                for key, value in node.items():
                                    if key == heading and isinstance(value, dict) and 'progress' in value:
                                        old_progress = value['progress']
                                        # ADD the progress increment to existing progress
                                        value['progress'] = old_progress + progress_increment
                                        print(f"  ✓ '{heading}': {old_progress} + {progress_increment} = {value['progress']}")
                                        return True
                                    elif isinstance(value, dict):
                                        if update_heading(value, heading, progress_increment):
                                            return True
                            return False
                        
                        # Helper function to find parent depth 0 topic for a heading
                        def find_parent_depth0(node, heading, parent_depth0=None):
                            """Recursively find the depth 0 topic that contains this heading"""
                            if not isinstance(node, dict):
                                return None
                            
                            # Traverse the structure
                            for key, value in node.items():
                                if not isinstance(value, dict):
                                    continue
                                
                                # Check if this value is a depth 0 node
                                if value.get('depth') == 0:
                                    # This is a depth 0 node - check if heading is in its contents
                                    contents = value.get('contents', {})
                                    if heading in contents:
                                        # Found the heading! Return this depth 0 topic's name
                                        return key
                                    # Search recursively inside this depth 0 node
                                    result = find_parent_depth0(value, heading, key)
                                    if result:
                                        return result
                                else:
                                    # Not a depth 0 node - check if this key is the heading we're looking for
                                    if key == heading and 'progress' in value:
                                        # Found the heading - return the parent depth 0 we've been tracking
                                        return parent_depth0
                                    # Continue searching recursively
                                    result = find_parent_depth0(value, heading, parent_depth0)
                                    if result:
                                        return result
                            
                            return None
                        
                        # Update each heading in the skill map
                        for heading, tier_str in progress_dict.items():
                            # Convert tier string to progress increment
                            progress_increment = tier_to_progress.get(tier_str, 0)
                            
                            if progress_increment > 0:
                                print(f"📊 Updating '{heading}' with Tier {tier_str} (+{progress_increment} points)")
                                update_heading(skill_map_dict, heading, progress_increment)
                                
                                # Find which depth 0 topic contains this heading
                                parent_topic = find_parent_depth0(skill_map_dict, heading)
                                if parent_topic:
                                    print(f"📊 Found parent depth 0 topic '{parent_topic}' for heading '{heading}'")
                                    updated_depth0_topics.add(parent_topic)
                                else:
                                    print(f"⚠️  WARNING: Could not find parent depth 0 topic for heading '{heading}'")
                            else:
                                print(f"⚠️  WARNING: Invalid tier value '{tier_str}' for heading '{heading}'")
                    
                    # Recalculate progress upward (average from bottom to top)
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
                    
                    # Calculate overall skill map progress
                    # print(f'skill map dict 2452 {skill_map_dict}')
                    calculate_progress_upward(skill_map_dict)
                    
                    # Convert back to JSON string
                    current_skill_map = json.dumps(skill_map_dict, indent=2)
                    print(f"✓ Progress recalculated upward")
                    
                    # Update MongoDB
                    if users_collection is not None:
                        thread_id = state.get('thread_id')
                        user_id = state.get('user_id')
                        
                        if thread_id and user_id:
                            users_collection.update_one(
                                {"user_id": user_id},
                                {"$set": {f"threads.{thread_id}.skill_map": skill_map_dict}},
                                upsert=True
                            )
                            print(f"✓ MongoDB updated: user={user_id}, thread={thread_id}")
                        else:
                            print(f"⚠ Missing IDs - user_id: {user_id}, thread_id: {thread_id}")
                    
                    print("Skill map update complete")
                    
                    # Add updated depth 0 topics to be passed to frontend
                    if updated_depth0_topics:
                        print(f"📊 Updated depth 0 topics: {list(updated_depth0_topics)}")
                    else:
                        print("⚠️  No valid progress dict found or wrong format")
                except Exception as e:
                    print(f"❌ Error parsing progress tracker output: {e}")
                    print(f"   Output was: {progress_output}")
            else:
                print("No progress to update (null output)")
                
        except Exception as e:
            print(f"Error in progress_tracker: {e}")

    if not ai_response_content:
        ai_response_content = ""

    # Format follow-ups as markdown
    # If block is True, don't add follow-up questions
    if updated_block:
        follow_up_questions = None
        final_content = ai_response_content
    elif follow_up_questions and isinstance(follow_up_questions, list):
        followups_md = "\n\n\n\n**Related:**\n" + "\n".join([f"- {q}" for q in follow_up_questions])
        final_content = ai_response_content + followups_md
    else:
        final_content = last_ai_msg.content
    print(f"Final content from node: {final_content}+{follow_up_questions}")
    print(f"DEBUG - Returning skill_map length: {len(current_skill_map) if current_skill_map else 0}")
    print(f"DEBUG - Returning chat_title: {updated_chat_title}")
    print(f"DEBUG - Returning block: {updated_block}")
    
    # Build additional_kwargs with follow_up_questions, quiz_available (chat), and sidebar_quizzes
    # quiz_available: shown in chat only if matches current_topic
    # sidebar_quizzes: shown in sidebar for other ready quizzes
    additional_kwargs = {"follow_up_questions": follow_up_questions, "block": updated_block}
    print(f"🔒 Setting block in additional_kwargs: {updated_block}")
    print(f"🔍 DEBUG quiz_available check: current_topic='{current_topic}', quiz_available={quiz_available}")
    if quiz_available:
        additional_kwargs["quiz_available"] = quiz_available
        print(f"✅ Quiz available for chat: '{quiz_available.get('topic', 'unknown')}' (progress: {quiz_available.get('progress', 0)})")
        print(f"   This quiz button should appear below related questions in the chat!")
    else:
        print(f"⚠️ No quiz_available set for chat (current_topic: '{current_topic}')")
    if sidebar_quizzes:
        additional_kwargs["sidebar_quizzes"] = sidebar_quizzes
        print(f"✅ Sidebar quizzes: {[q['topic'] for q in sidebar_quizzes]}")
    
    # Add updated depth 0 topics for progress notifications
    if updated_depth0_topics:
        additional_kwargs["updated_topics"] = list(updated_depth0_topics)
        print(f"📊 Added updated_topics to additional_kwargs: {list(updated_depth0_topics)}")
    
    return {
        "messages": [AIMessage(content=final_content, additional_kwargs=additional_kwargs)], 
        "skill_map": current_skill_map,  # Use updated skill map with new progress
        "chat_title": updated_chat_title,
        "current_topic": current_topic,  # Update current_topic in state
        "block": updated_block,  # Update block in state
        "skill": state.get("skill", ""),  # Preserve skill
        "user_goal": state.get("user_goal", ""),  # Preserve user_goal
        "quiz_data": state.get("quiz_data", [])  # Preserve quiz_data
    }




# graph builder
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("educational_node", educational_node)

    # Single-node loop: generate when last is human, interrupt when last is AI
    graph.add_edge(START, "educational_node")
    graph.add_edge("educational_node", "educational_node")

    return graph.compile(checkpointer=memory)

graph_quiz = build_graph()