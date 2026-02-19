import os
import uuid
import time
import random
from typing import List, Dict, Any

import pymongo
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

# from graph import graph, memory, store_chats, store_skill_map, update_quiz_answers
from graph_quiz import graph_quiz as graph, memory, store_chats, store_skill_map, update_quiz_answers
from langgraph.types import Command


from fastapi.responses import StreamingResponse
import asyncio
import json


def parse_skill_map_to_progress(skill_map_text: str) -> dict:
    """
    Parse skill map text into progress tracking structure.
    Returns dict with domains, topics, subtopics, and progress initialized to 0.
    """
    if not skill_map_text or skill_map_text.strip() == "":
        return {"domains": []}
    
    try:
        domains = []
        current_domain = None
        current_topic = None
        
        lines = skill_map_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Domain line (starts with "Domain:")
            if line.startswith("Domain:"):
                if current_domain:  # Save previous domain
                    domains.append(current_domain)
                current_domain = {
                    "domain_name": line.replace("Domain:", "").strip(),
                    "topics": []
                }
                current_topic = None
            
            # Topic line (starts with "Topic:")
            elif line.startswith("Topic:"):
                if current_domain and current_topic:  # Save previous topic
                    current_domain["topics"].append(current_topic)
                current_topic = {
                    "topic_name": line.replace("Topic:", "").strip(),
                    "progress": 0,
                    "subtopics": {}
                }
            
            # Subtopic line (starts with "- ")
            elif line.startswith("-") and current_topic:
                subtopic = line.lstrip("- ").strip()
                if subtopic:
                    current_topic["subtopics"][subtopic] = 0
        
        # Save last topic and domain
        if current_topic and current_domain:
            current_domain["topics"].append(current_topic)
        if current_domain:
            domains.append(current_domain)
        
        return {"domains": domains}
    
    except Exception as e:
        print(f"Error parsing skill map: {e}")
        return {"domains": []}


# Load environment variables
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Chatbot API",
    description="API for the LangGraph Guardrails Chatbot",
    version="1.0.0",
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3001",
        "http://localhost:3002", 
        "http://127.0.0.1:3002",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],  # Frontend URLs (Next.js can run on different ports)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["X-Thread-ID"],  # Expose custom headers to frontend
)

# --- MongoDB Connection ---
mongo_url = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = pymongo.MongoClient(mongo_url)
db = client.get_database("chatbot")
checkpoints_collection = db.get_collection("checkpoints")
chats_collection = db.get_collection("chats")
users_collection = db.get_collection("users")

## --- Pydantic Models ---
class SignupRequest(BaseModel):
    name: str
    email: str
    phone: str

class SignupResponse(BaseModel):
    user_id: str
    name: str
    email: str
    phone: str
    message: str

class LoginRequest(BaseModel):
    email: str

class LoginResponse(BaseModel):
    user_id: str
    name: str
    email: str
    phone: str
    threads: dict
    message: str

class StartChatRequest(BaseModel):
    message: str
    user_id: str

class StartChatResponse(BaseModel):
    thread_id: str
    response: str

class ResumeChatRequest(BaseModel):
    thread_id: str
    message: str
    user_id: str

class TransferThreadRequest(BaseModel):
    temp_user_id: str
    real_user_id: str
    thread_id: str

class QuizDataRequest(BaseModel):
    user_id: str

class QuizDataResponse(BaseModel):
    user_id: str
    quiz_data: list

class ChatMessage(BaseModel):
    role: str
    content: str
    quiz: dict | None = None  # Optional field for quiz messages
    quiz_available: dict | None = None  # Optional field for quiz availability (chat button)
    sidebar_quizzes: list[dict] | None = None  # Optional field for sidebar quiz buttons
    follow_up_questions: list[str] | None = None  # Optional field for AI messages
    # NOTE: block flag is NOT stored in DB - handled only in response/SSE stream and localStorage

class HealthCheck(BaseModel):
    status: str
    message: str

class QuizAnswerUpdate(BaseModel):
    question_index: int
    user_answer: int

class UpdateQuizAnswersRequest(BaseModel):
    thread_id: str
    user_id: str
    quiz_id: str
    topic: str
    answers: list[QuizAnswerUpdate]

class UpdateQuizAnswersResponse(BaseModel):
    topic: str
    score: int
    total_questions: int
    status: str  # passed, failed, unattempted
    message: str

# --- API Endpoints ---
@app.get("/health", response_model=HealthCheck)
def health_check():
    """
    Health check endpoint to confirm the API is running.
    """
    return {"status": "healthy", "message": "Chatbot API is running"}

# --- User Management Endpoints ---
@app.post("/signup", response_model=SignupResponse)
def signup(request: SignupRequest):
    """
    Sign up a new user. Creates a new user document with unique user_id.
    """
    try:
        # Check if email already exists
        existing_user = users_collection.find_one({"email": request.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Generate unique user_id
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        # Create new user document
        user_doc = {
            "user_id": user_id,
            "name": request.name,
            "email": request.email,
            "phone": request.phone,
            "threads": {},  # Empty dict to store thread_ids
            "created_at": str(uuid.uuid4().hex[:8])
        }
        
        users_collection.insert_one(user_doc)
        
        return SignupResponse(
            user_id=user_id,
            name=request.name,
            email=request.email,
            phone=request.phone,
            message="User created successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
def login(request: LoginRequest):
    """
    Login existing user. Returns user_id and threads.
    If user not found, returns signup_required flag.
    """
    try:
        # Find user by email
        user = users_collection.find_one({"email": request.email})
        
        if not user:
            # Instead of throwing error, return signup_required flag
            return {
                "signup_required": True,
                "email": request.email,
                "message": "User not found. Please sign up."
            }
        
        return LoginResponse(
            user_id=user["user_id"],
            name=user.get("name", ""),
            email=user.get("email", request.email),
            phone=user.get("phone", ""),
            threads=user.get("threads", {}),
            message="Login successful"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/threads")
def get_user_threads(user_id: str):
    """
    Get all threads for a specific user with details from chats collection.
    """
    try:
        user = users_collection.find_one({"user_id": user_id})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        thread_ids = list(user.get("threads", {}).keys())
        conversations = []
        
        # Get details for each thread from chats collection
        for thread_id in thread_ids:
            chat = chats_collection.find_one({"thread_id": thread_id})
            if chat and "messages" in chat and len(chat["messages"]) > 0:
                # Use stored title if available, otherwise fallback to first message
                if "title" in chat and chat["title"]:
                    title = chat["title"]
                else:
                    # Fallback: Get first user message as title
                    first_message = chat["messages"][0]
                    title = first_message.get("content", "New Chat")[:50]
                    if len(first_message.get("content", "")) > 50:
                        title += "..."
                
                conversations.append({
                    "thread_id": thread_id,
                    "title": title,
                    "date": chat.get("timestamp", "")
                })
        
        return {"user_id": user_id, "conversations": conversations}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quizdata/{user_id}", response_model=list)
def get_quiz_data(user_id: str):
    """
    Get Quiz Data for a user_id and thread_id. Return the quiz_data as a list.
    """

    try:
        user=users_collection.find_one({"user_id": user_id})
        print(f'game of thrones user: {user}')
        #people management
        #HR2
        user_quiz_data = [
 {'quiz_id': 1,
  'user_answer': 'Build trust and psychological safety',
  'is_correct': 1,
  'question': 'What is the foundation of effective people management?',
  'option_a': 'Authority and control',
  'option_b': 'Clear hierarchy',
  'option_c': 'Build trust and psychological safety',
  'option_d': 'Strict performance monitoring',
  'correct_option': 'c',
  'topic_tested': 'Foundations of People Management'},

 {'quiz_id': 2,
  'user_answer': 'Publicly call out poor performance',
  'is_correct': 0,
  'question': 'How should managers address underperformance?',
  'option_a': 'Publicly call out poor performance',
  'option_b': 'Provide private, constructive feedback',
  'option_c': 'Ignore it temporarily',
  'option_d': 'Escalate immediately',
  'correct_option': 'b',
  'topic_tested': 'Performance Feedback'},

 {'quiz_id': 3,
  'user_answer': 'Understand individual motivations and goals',
  'is_correct': 1,
  'question': 'What enables managers to motivate diverse team members effectively?',
  'option_a': 'Uniform incentives',
  'option_b': 'Understand individual motivations and goals',
  'option_c': 'Strict goal enforcement',
  'option_d': 'Competition between employees',
  'correct_option': 'b',
  'topic_tested': 'Employee Motivation'},

 {'quiz_id': 4,
  'user_answer': 'Address issues early and directly',
  'is_correct': 1,
  'question': 'What is the best approach to managing interpersonal conflict?',
  'option_a': 'Avoid conflict entirely',
  'option_b': 'Address issues early and directly',
  'option_c': 'Let team members resolve it themselves',
  'option_d': 'Separate conflicting individuals',
  'correct_option': 'b',
  'topic_tested': 'Conflict Resolution'},

 {'quiz_id': 5,
  'user_answer': 'Treat everyone exactly the same',
  'is_correct': 0,
  'question': 'Why can treating everyone exactly the same be ineffective?',
  'option_a': 'It increases bias',
  'option_b': 'It ignores individual needs and contexts',
  'option_c': 'It reduces fairness',
  'option_d': 'It lowers accountability',
  'correct_option': 'b',
  'topic_tested': 'Equity vs Equality'},

 {'quiz_id': 6,
  'user_answer': 'Develop others through coaching and feedback',
  'is_correct': 1,
  'question': 'What distinguishes strong people managers from average ones?',
  'option_a': 'Technical expertise',
  'option_b': 'Develop others through coaching and feedback',
  'option_c': 'Strict control over outcomes',
  'option_d': 'Delegating all responsibility',
  'correct_option': 'b',
  'topic_tested': 'Coaching and Development'},

 {'quiz_id': 7,
  'user_answer': 'Set clear expectations and follow through consistently',
  'is_correct': 1,
  'question': 'How do managers build credibility with their teams?',
  'option_a': 'Being flexible with rules',
  'option_b': 'Set clear expectations and follow through consistently',
  'option_c': 'Avoid difficult conversations',
  'option_d': 'Rely on senior leadership authority',
  'correct_option': 'b',
  'topic_tested': 'Manager Credibility'},

 {'quiz_id': 8,
  'user_answer': 'Focus only on results',
  'is_correct': 0,
  'question': 'What is a risk of focusing only on results and ignoring behavior?',
  'option_a': 'Higher productivity',
  'option_b': 'Short-term gains at the cost of long-term team health',
  'option_c': 'Clear accountability',
  'option_d': 'Reduced ambiguity',
  'correct_option': 'b',
  'topic_tested': 'Sustainable Team Performance'},

 {'quiz_id': 9,
  'user_answer': 'Adapt leadership style to team maturity',
  'is_correct': 1,
  'question': 'How should leadership style evolve as teams mature?',
  'option_a': 'Remain directive',
  'option_b': 'Adapt leadership style to team maturity',
  'option_c': 'Increase control',
  'option_d': 'Reduce communication',
  'correct_option': 'b',
  'topic_tested': 'Situational Leadership'},

 {'quiz_id': 10,
  'user_answer': 'Avoid giving feedback unless asked',
  'is_correct': 0,
  'question': 'Which behavior most limits team growth?',
  'option_a': 'Avoid giving feedback unless asked',
  'option_b': 'Continuous learning',
  'option_c': 'Open communication',
  'option_d': 'Shared accountability',
  'correct_option': 'a',
  'topic_tested': 'Feedback Culture'}
]





        return QuizDataResponse(
            user_id=user["user_id"],
            quiz_data=user_quiz_data
        )

    except HTTPException:
        raise

# --- Event Generator for Streaming ---
async def event_generator(thread_id: str, initial_message: str = None, is_new_chat: bool = False, user_id: str = None):
    """Generator for SSE streaming"""


    quiz_data = get_quiz_data(user_id).quiz_data
    # print(f'output from quiz data endpoint: {get_quiz_data(user_id).quiz_data}')
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id, "quiz_data": quiz_data}}
    
    try:
        streamed_already = False
        
        #static block
        if is_new_chat and initial_message:
            message_lower = initial_message.strip().lower()
            start_words = ["hi", "hey", "hello", "yo", "sup"]
            
            words = initial_message.strip().split()
            if len(words) == 1:
                static_block = False
                for x in start_words:
                    if x in message_lower:
                        static_block = True
                        break
                
                if static_block:
                    related_questions = [
                        "I want to learn Python and make real projects",
                        "I want to learn how to build AI tools",
                        "I want to learn about product management",
                        "I want to learn UI/UX and design apps",
                        "I want to get into data science + analytics",
                        "How to learn coding",
                        "How to level up my speaking + communication skills",
                        "How to master digital marketing and grow a brand",
                        "How to become a finance analyst",
                        "How to become a content creator"
                    ]
                    
                    selected_questions = random.sample(related_questions, 3)
                    
                    static_response = """Hey there! Welcome to Threads. 
                    
I'm your AI Teaching Buddy from LearnTube.ai — think ChatGPT, but built ONLY for mastering new skills.

I don't just answer your questions…

I build your entire learning path. I track your progress. I quiz you. I turn you into a pro — step by step.

You pick a skill. I turn it into a personalized roadmap. You just learn.

**Related:**

- {question1}
- {question2}
- {question3}""".format(
                        question1=selected_questions[0],
                        question2=selected_questions[1],
                        question3=selected_questions[2]
                    )
                    
                    ai_message = AIMessage(
                        content=static_response,
                        additional_kwargs={
                            "follow_up_questions": selected_questions
                        }
                    )
                    
                    store_chats(thread_id, [ai_message])
                    
                    # CRITICAL: Initialize state in checkpoint even for static response
                    # This ensures skill_map, title, current_topic are available for resume
                    # We invoke the graph with the state including the AI message
                    # The educational_node will see the AI message and just interrupt (no processing)
                    initial_state_input = {
                        "messages": [HumanMessage(content=initial_message), ai_message],
                        "skill_map": "",
                        "chat_title": "",
                        # "title": "",
                        "thread_id": thread_id,
                        "user_id": user_id,
                        "current_topic": "",
                        "block": False,
                        "quiz_data": quiz_data,
                        "user_goal": "",
                        "skill": "",
                    }
                    
                    # Initialize checkpoint by invoking graph (educational_node will see AI message and interrupt)
                    try:
                        graph.invoke(initial_state_input, config=config)
                        print(f"✅ Initialized checkpoint state for thread {thread_id} with static greeting")
                    except Exception as e:
                        print(f"⚠️ Failed to initialize checkpoint state: {e}")
                    
                    yield f"data: {json.dumps({'type': 'start'})}\n\n"
                    
                    current = ""
                    for char in static_response:
                        current += char
                        yield f"data: {json.dumps({'content': current})}\n\n"
                        await asyncio.sleep(0.02)
                    
                    yield f"data: {json.dumps({'type': 'followups', 'items': selected_questions})}\n\n"
                    
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
        
        # Create input based on whether it's a new chat or resuming
        if is_new_chat:
            input_data = {
                    "messages": [HumanMessage(content=initial_message)],
                    "skill_map": "",
                    "chat_title": "",
                    # "title": "",
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "current_topic": "",
                    "block": False,
                    "quiz_data": quiz_data,
                    "user_goal": "",
                    "skill": "",
                }
        else:
            input_data = {
                "messages": [HumanMessage(content=initial_message)],
                "thread_id": thread_id,
                "user_id": user_id,
                "skill_map": "",
                "current_topic": "",
                "block": False,
                "quiz_data": quiz_data,
                "user_goal": "",
                "skill": "",
                # "title": "",
            }
        
        # Stream from LangGraph
        if is_new_chat:
            stream_source = graph.stream(input_data, config=config, stream_mode="updates")
            # print(f"stream source from endpoints 334: {stream_source}")
        else:
            # For resume, we need to load skill_map and chat_title from MongoDB
            # CRITICAL: chat_title must be extracted from skill_map's title field
            skill_map_str = ""
            chat_title_from_db = ""
            
            if user_id and thread_id:
                try:
                    user_doc = users_collection.find_one({"user_id": user_id})
                    if user_doc:
                        thread_data = user_doc.get("threads", {}).get(thread_id, {})
                        skill_map_data = thread_data.get("skill_map")
                        
                        if skill_map_data:
                            # If skill_map is a dict, convert to JSON string
                            if isinstance(skill_map_data, dict):
                                skill_map_str = json.dumps(skill_map_data)
                                # Extract title from skill_map's title field
                                chat_title_from_db = skill_map_data.get("title", "")
                            else:
                                skill_map_str = skill_map_data
                                # Try to parse and extract title
                                try:
                                    skill_map_parsed = json.loads(skill_map_str)
                                    chat_title_from_db = skill_map_parsed.get("title", "")
                                except:
                                    pass
                            
                            print(f"✅ Loaded skill_map and chat_title from DB: chat_title='{chat_title_from_db}'")
                        
                        # Load skill and user_goal from thread_data
                        user_goal_from_db = thread_data.get("user_goal", "")
                        skill_from_db = thread_data.get("skill", "")
                        if user_goal_from_db:
                            print(f"✅ Loaded user_goal from DB: '{user_goal_from_db}'")
                        if skill_from_db:
                            print(f"✅ Loaded skill from DB: '{skill_from_db}'")
                            
                except Exception as e:
                    print(f"⚠️ Failed to load skill_map/chat_title from DB: {e}")
            
            # For resume, we need to update user_id in state to handle login after temp user
            # Pass the message and user_id to update state - LangGraph will merge with checkpointed state
            # This ensures user_id is always current (not temp_user) when resuming after login
            resume_input = {
                "messages": [HumanMessage(content=initial_message)],
                "skill_map": skill_map_str,
                "chat_title": chat_title_from_db,
                # "title": "",
                "thread_id": thread_id,
                "user_id": user_id,
                "current_topic": "",
                "block": False,
                "quiz_data": quiz_data,
                "user_goal": user_goal_from_db if 'user_goal_from_db' in locals() else "",
                "skill": skill_from_db if 'skill_from_db' in locals() else "",
            }
            if user_id:
                # CRITICAL: Update user_id in state to override any temp_user_id from checkpoint
                # This ensures all operations (quiz storage, etc.) use the logged-in user_id
                resume_input["user_id"] = user_id
                print(f"🔄 Resuming chat with updated user_id: {user_id} (overriding any temp_user_id from checkpoint)")
            stream_source = graph.stream(
                resume_input,
                config=config,
                stream_mode="updates"
            )

        chat_title_from_state = None
        skill_map_from_state = None
        
        for chunk in stream_source:
            # Loop through all nodes in chunk
            for node_name, node_output in chunk.items():
                if not node_output or not isinstance(node_output, dict):
                    continue
                
                # Extract chat_title if present in state update
                if "chat_title" in node_output and node_output["chat_title"]:
                    chat_title_from_state = node_output["chat_title"]
                    print(f"Chat title extracted from state: {chat_title_from_state}")
                
                # Extract skill_map if present in state update (allow empty check to fail gracefully)
                if "skill_map" in node_output:
                    skill_map_from_state = node_output["skill_map"]
                    # Only store if not empty
                    if skill_map_from_state and skill_map_from_state.strip():
                        # print(f"Skill map extracted from state: {skill_map_from_state[:100]}...")
                        
                        # CRITICAL: Extract chat_title from skill_map's title field if not already set
                        if not chat_title_from_state:
                            try:
                                # Parse skill_map to extract title
                                if isinstance(skill_map_from_state, str):
                                    skill_map_parsed = json.loads(skill_map_from_state)
                                else:
                                    skill_map_parsed = skill_map_from_state
                                
                                # Extract title from skill_map and use as chat_title
                                title_from_skill_map = skill_map_parsed.get("title", "")
                                if title_from_skill_map:
                                    chat_title_from_state = title_from_skill_map
                                    print(f"✅ Extracted chat_title from skill_map: '{chat_title_from_state}'")
                            except Exception as e:
                                print(f"⚠️ Failed to extract chat_title from skill_map: {e}")
                        
                        # Store skill map immediately when detected
                        if user_id and thread_id:
                            store_skill_map(thread_id, user_id, skill_map_from_state, chat_title_from_state or "Learning Path")
                    else:
                        print(f"⚠ Skill map in node_output but empty: {skill_map_from_state}")
                    
                # Check if this node updated messages
                if "messages" in node_output:
                    messages = node_output["messages"]
                    if not messages:
                        continue
                    
                    # Get last message
                    last_msg = messages[-1] if isinstance(messages, list) else messages
                    
                    # Stream only AI messages and only once
                    if hasattr(last_msg, "type") and last_msg.type == "ai" and not streamed_already:
                        content = last_msg.content
                        
                        # Send start signal
                        yield f"data: {json.dumps({'type': 'start'})}\n\n"
                        
                        # Stream character by character
                        current = ""
                        for char in content:
                            current += char
                            yield f"data: {json.dumps({'content': current})}\n\n"
                            await asyncio.sleep(0.02)
                        
                        # After full text is streamed, send follow-up questions and quiz availability if available
                        try:
                            followups = []
                            quiz_available = None
                            sidebar_quizzes = None
                            block = False
                            updated_topics = None
                            if hasattr(last_msg, 'additional_kwargs') and isinstance(last_msg.additional_kwargs, dict):
                                fu = last_msg.additional_kwargs.get('follow_up_questions')
                                if isinstance(fu, list):
                                    followups = fu
                                
                                # Extract quiz_available (for chat button)
                                quiz_available = last_msg.additional_kwargs.get('quiz_available')
                                
                                # Extract sidebar_quizzes (for sidebar buttons)
                                sidebar_quizzes = last_msg.additional_kwargs.get('sidebar_quizzes')
                                
                                # Extract block flag
                                block = last_msg.additional_kwargs.get('block', False)
                                
                                # Extract updated_topics (for progress notifications)
                                updated_topics = last_msg.additional_kwargs.get('updated_topics')
                            
                            if followups:
                                yield f"data: {json.dumps({'type': 'followups', 'items': followups})}\n\n"
                            
                            # Send block flag if present
                            if block:
                                print(f"🔒 Sending block flag to frontend: {block}")
                                yield f"data: {json.dumps({'type': 'block', 'data': True})}\n\n"
                            
                            # Send quiz_available if present (for chat button)
                            if quiz_available:
                                print(f"🎯 Sending quiz_available to frontend: {quiz_available}")
                                yield f"data: {json.dumps({'type': 'quiz_available', 'data': quiz_available})}\n\n"
                            
                            # Send sidebar_quizzes if present (for sidebar buttons)
                            if sidebar_quizzes:
                                print(f"🎯 Sending sidebar_quizzes to frontend: {sidebar_quizzes}")
                                yield f"data: {json.dumps({'type': 'sidebar_quizzes', 'data': sidebar_quizzes})}\n\n"
                            
                            # Send updated_topics if present (for progress notifications)
                            if updated_topics and isinstance(updated_topics, list) and len(updated_topics) > 0:
                                print(f"📊 Sending updated_topics to frontend: {updated_topics}")
                                yield f"data: {json.dumps({'type': 'updated_topics', 'data': updated_topics})}\n\n"
                            
                            # Check if this is a quiz (when user clicked "Take quiz for")
                            if hasattr(last_msg, 'additional_kwargs') and isinstance(last_msg.additional_kwargs, dict):
                                quiz_data = last_msg.additional_kwargs.get('quiz')
                                if quiz_data:
                                    print(f"📝 Sending quiz data to frontend: {quiz_data['topic']} with {len(quiz_data['questions'])} questions")
                                    yield f"data: {json.dumps({'type': 'quiz', 'data': quiz_data})}\n\n"
                        except Exception as e:
                            print(f"Failed to emit followups/quiz: {e}")

                        # Store only the NEW AI message (user message already stored in endpoint)
                        store_chats(thread_id, [last_msg])
                        
                        # Store chat title in chats collection (skill_map already stored by store_skill_map in graph.py)
                        if chat_title_from_state:
                            chats_collection.update_one(
                                {"thread_id": thread_id},
                                {"$set": {"title": chat_title_from_state}},
                                upsert=True
                            )
                            print(f"Chat title stored in chats collection: {chat_title_from_state}")
                        
                        streamed_already = True
        
        # After streaming completes, check final state for skill_map and store if not already stored
        final_state = graph.get_state(config)
        if final_state and final_state.values:
            final_skill_map = final_state.values.get("skill_map")
            final_chat_title = final_state.values.get("chat_title")
            
            # CRITICAL: Extract chat_title from skill_map's title field if not already set
            if final_skill_map and not final_chat_title:
                try:
                    # Parse skill_map to extract title
                    if isinstance(final_skill_map, str):
                        skill_map_parsed = json.loads(final_skill_map)
                    else:
                        skill_map_parsed = final_skill_map
                    
                    # Extract title from skill_map and use as chat_title
                    title_from_skill_map = skill_map_parsed.get("title", "")
                    if title_from_skill_map:
                        final_chat_title = title_from_skill_map
                        print(f"✅ Extracted chat_title from final skill_map: '{final_chat_title}'")
                except Exception as e:
                    print(f"⚠️ Failed to extract chat_title from final skill_map: {e}")
            
            # Store skill_map if it exists and wasn't stored during streaming
            if final_skill_map and not skill_map_from_state and user_id and thread_id:
                print(f"Storing skill_map from final state")
                store_skill_map(thread_id, user_id, final_skill_map, final_chat_title or "Learning Path")
        
        # Done
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/start-chat")
async def start_chat(request: StartChatRequest):
    """
    Starts a new chat session with streaming response.
    Links the thread to the user_id.
    """
    thread_id = f"thread_{uuid.uuid4().hex[:8]}"
    
    try:
        # Add thread_id to user's threads dict (initialize as empty object)
        # Use upsert=True to create temp user if they don't exist
        users_collection.update_one(
            {"user_id": request.user_id},
            {"$set": {f"threads.{thread_id}": {}}},  # Initialize empty object for thread data
            upsert=True  # Create user document if it doesn't exist (for temp users)
        )
        
        # Store initial user message immediately so chat history endpoint works
        initial_user_message = HumanMessage(content=request.message)
        store_chats(thread_id, [initial_user_message])
        
        # Return streaming response with thread_id in headers
        response = StreamingResponse(
            event_generator(thread_id, request.message, is_new_chat=True, user_id=request.user_id),
            media_type="text/event-stream",
            headers={
                "X-Thread-ID": thread_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resume-chat")
async def resume_chat(request: ResumeChatRequest):
    """
    Resumes an existing chat session with streaming response.
    """
    
    try:
        # Store the user message immediately for resume-chat too
        user_message = HumanMessage(content=request.message)
        store_chats(request.thread_id, [user_message])
        
        return StreamingResponse(
            event_generator(request.thread_id, request.message, is_new_chat=False, user_id=request.user_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def fetch_chat_history(thread_id: str) -> List[Dict[str, str]]:
    """
    Fetches chat history from the chats collection for a given thread_id.
    """
    if chats_collection is None:
        return []

    document = chats_collection.find_one({"thread_id": thread_id})
    if document and "messages" in document:
        return document["messages"]
    return []

@app.get("/chat-history/{thread_id}", response_model=List[ChatMessage])
def get_chat_history(thread_id: str):
    """
    Retrieves the chat history for a specific thread ID.
    Returns empty list if no history found (instead of 404 error).
    """
    try:
        history = fetch_chat_history(thread_id)
        if not history:
            # Return empty list instead of 404 error for new threads
            return []
        
        # CRITICAL FIX: Include follow_up_questions, quiz_available, sidebar_quizzes when present
        # NOTE: block flag is NOT included here - it's handled only in SSE stream and localStorage
        messages = []
        for m in history:
            if "quiz" in m.keys():
                msg = ChatMessage(
                    role=m["role"], 
                    content="",
                    quiz=m.get("quiz"),
                    quiz_available=m.get("quiz_available"),  # Include if present
                    sidebar_quizzes=m.get("sidebar_quizzes"),  # Include if present
                    follow_up_questions=m.get("follow_up_questions")  # Include if present
                )
            else:
                msg = ChatMessage(
                    role=m["role"], 
                    content=m["content"],
                    quiz_available=m.get("quiz_available"),  # Include if present
                    sidebar_quizzes=m.get("sidebar_quizzes"),  # Include if present
                    follow_up_questions=m.get("follow_up_questions")  # Include if present
                )
            messages.append(msg)
        

        return messages
    except Exception as e:
        # Log the error but return empty list to prevent frontend crashes
        print(f"Error fetching chat history for {thread_id}: {e}")
        return []

@app.get("/progress/{user_id}/{thread_id}")
def get_progress(user_id: str, thread_id: str):
    """
    Get progress data for a specific user and thread.
    Returns domains with topics and their progress values.
    """
    try:
        user = users_collection.find_one({"user_id": user_id})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get progress data for the specific thread
        thread_data = user.get("threads", {}).get(thread_id, {})
        progress_data = thread_data.get("progress_data", {"domains": []})
        
        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "progress_data": progress_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/skill-map/{user_id}/{thread_id}")
def get_skill_map(user_id: str, thread_id: str):
    """
    Get skill map for a specific user and thread.
    Returns the hierarchical skill map with depth-based structure.
    """
    try:
        
        user = users_collection.find_one({"user_id": user_id})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get skill map for the specific thread
        thread_data = user.get("threads", {}).get(thread_id, {})
        
        if not thread_data or "skill_map" not in thread_data:
            raise HTTPException(status_code=404, detail="Skill map not found for this thread")
        
        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "chat_title": thread_data.get("chat_title", "Learning Path"),
            "skill_map": thread_data.get("skill_map"),
            "updated_at": thread_data.get("updated_at", time.time())
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/skill-map-status/{user_id}/{thread_id}")
def get_skill_map_status(user_id: str, thread_id: str):
    """
    Get skill map status and data for polling.
    Returns status (building/success/fail) and skill map data if available.
    """
    try:
        user = users_collection.find_one({"user_id": user_id})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        thread_data = user.get("threads", {}).get(thread_id, {})
        
        if not thread_data:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Only return "building" if the field actually exists and is set to "building"
        # Otherwise return "none" to indicate skill map generation hasn't started
        status = thread_data.get("skill_map_status", "none")
        skill_map = thread_data.get("skill_map", None)
        
        return {
            "status": status,
            "data": skill_map
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-quiz-answers", response_model=UpdateQuizAnswersResponse)
def update_quiz_answers_endpoint(request: UpdateQuizAnswersRequest):
    """
    Update quiz answers and calculate score.
    
    Request body:
    {
        "thread_id": "thread_123",
        "user_id": "user_456",
        "quiz_id": "quiz_123",
        "topic": "Variables and Data Types",
        "answers": [
            {"question_index": 0, "user_answer": 2},
            {"question_index": 1, "user_answer": 0},
            ...
        ]
    }
    """
    try:
        print(f"Update quiz request - user_id: {request.user_id}, thread_id: {request.thread_id}, quiz_id: {request.quiz_id}, topic: '{request.topic}'")
        print(f"Answers count: {len(request.answers)}")
        
        # Convert Pydantic models to dict format
        answers_list = [{"question_index": a.question_index, "user_answer": a.user_answer} for a in request.answers]
        
        # Debug: Check if user and quiz exist before updating
        from graph import users_collection
        if users_collection is not None:
            user_doc = users_collection.find_one({"user_id": request.user_id})
            if user_doc:
                threads = user_doc.get("threads", {})
                if request.thread_id in threads:
                    quizzes = threads[request.thread_id].get("quizzes", {})
                    print(f"📝 Available quiz topics for this thread: {list(quizzes.keys())}")
                    if request.topic in quizzes:
                        topic_quizzes = quizzes[request.topic]
                        # Handle array structure - quizzes[topic] is now an array
                        if isinstance(topic_quizzes, list) and len(topic_quizzes) > 0:
                            # Find quiz by quiz_id in array
                            quiz_found = None
                            for quiz in topic_quizzes:
                                if isinstance(quiz, dict) and "quiz_id" in quiz and quiz["quiz_id"] == request.quiz_id:
                                    quiz_found = quiz
                                    break
                            if quiz_found and "questions" in quiz_found:
                                questions_count = len(quiz_found["questions"]) if isinstance(quiz_found["questions"], list) else 0
                                print(f"✅ Quiz found! Questions count: {questions_count}")
                            else:
                                print(f"⚠️ Quiz topic found but quiz_id '{request.quiz_id}' not found in array")
                        else:
                            print(f"⚠️ Quiz topic found but array is empty or invalid")
                    else:
                        print(f"❌ Quiz topic '{request.topic}' not found in quizzes. Available: {list(quizzes.keys())}")
                else:
                    print(f"❌ Thread {request.thread_id} not found in user's threads")
            else:
                print(f"❌ User {request.user_id} not found")
        
        # Update quiz answers
        print(f"updatequizanswersrequest request body from 907 {UpdateQuizAnswersRequest}")
        print(f"request from 907 endpoints: {request}")
        result = update_quiz_answers(
            request.thread_id,
            request.user_id,
            request.quiz_id,
            request.topic,
            answers_list
        )
        
        if not result:
            error_msg = f"Quiz not found or update failed. Check: user_id={request.user_id}, thread_id={request.thread_id}, topic='{request.topic}', quiz_id='{request.quiz_id}'"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
        
        print(f"✅ Quiz updated successfully - Score: {result['score']}/{result['total_questions']}, Status: {result['status']}")
        
        # CRITICAL: Include questions with user_answer fields in response
        response_data = UpdateQuizAnswersResponse(
            topic=result["topic"],
            score=result["score"],
            total_questions=result["total_questions"],
            status=result["status"],
            message=f"Quiz answers updated successfully. Score: {result['score']}/{result['total_questions']}"
        )
        
        # Add questions to response (convert to dict to include in JSON)
        response_dict = response_data.model_dump()
        if "questions" in result:
            response_dict["questions"] = result["questions"]
        
        return response_dict
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in update_quiz_answers_endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quiz/{user_id}/{thread_id}/{topic}/{quiz_id}")
def get_quiz(user_id: str, thread_id: str, topic: str, quiz_id: str):
    """
    Get quiz data for a specific user, thread, topic, and quiz_id.
    Returns the quiz with questions, user answers, and score.
    New architecture: quizzes.{topic}.{quiz_id} = quiz_data
    """
    try:
        user = users_collection.find_one({"user_id": user_id})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # New architecture: threads.{thread_id}.quizzes.{topic} = [quiz1, quiz2, ...]
        thread_data = user.get("threads", {}).get(thread_id, {})
        quizzes = thread_data.get("quizzes", {})
        
        if topic not in quizzes:
            raise HTTPException(status_code=404, detail=f"Quiz not found for topic: {topic}")
        
        topic_quizzes_raw = quizzes[topic]
        
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
            raise HTTPException(status_code=404, detail=f"Quiz data structure invalid for topic: {topic}")
        
        # Find quiz by quiz_id in the array
        quiz_data = None
        for quiz in topic_quizzes:
            if not isinstance(quiz, dict):
                continue
            if quiz.get("quiz_id") == quiz_id:
                quiz_data = quiz.copy()
                break
        
        if not quiz_data:
            raise HTTPException(status_code=404, detail=f"Quiz not found for topic: {topic} with quiz_id: {quiz_id}")
        
        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "topic": topic,
            "quiz": quiz_data,
            "quiz_id": quiz_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quizzes/{user_id}/{thread_id}")
def get_quiz_statuses(user_id: str, thread_id: str):
    """
    Get all quiz statuses for a specific user and thread.
    Returns a dict mapping topic -> status (passed/failed/unattempted)
    New architecture: quizzes.{topic} = [quiz1, quiz2, ...]
    Priority: passed > unattempted > failed (for determining topic status)
    """
    try:
        user = users_collection.find_one({"user_id": user_id})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # New architecture: threads.{thread_id}.quizzes.{topic} = [quiz1, quiz2, ...]
        thread_data = user.get("threads", {}).get(thread_id, {})
        quizzes = thread_data.get("quizzes", {})
        
        # Extract statuses - determine status based on all quizzes for that topic
        statuses = {}
        for topic_name, topic_quizzes_raw in quizzes.items():
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
                statuses[topic_name] = "unattempted"
                continue
            
            if not topic_quizzes:
                statuses[topic_name] = "unattempted"
                continue
            
            # Priority: passed > unattempted > failed
            topic_status = "unattempted"
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
            
            statuses[topic_name] = topic_status
        
        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "statuses": statuses
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transfer-thread")
async def transfer_thread(request: TransferThreadRequest):
    """
    Transfer a thread from temp_user_id to real_user_id.
    Moves all thread data (skill_map, chat_title, quizzes, etc.) from temp user to real user.
    """
    try:
        # Get the temp user document
        temp_user = users_collection.find_one({"user_id": request.temp_user_id})
        if not temp_user:
            raise HTTPException(status_code=404, detail=f"Temp user {request.temp_user_id} not found")
        
        threads = temp_user.get("threads", {})
        if request.thread_id not in threads:
            raise HTTPException(status_code=404, detail=f"Thread {request.thread_id} not found in temp user")
        
        # Get the thread data from temp user
        thread_data = threads[request.thread_id]
        
        # Transfer thread data to real user
        users_collection.update_one(
            {"user_id": request.real_user_id},
            {
                "$set": {
                    f"threads.{request.thread_id}": thread_data
                }
            },
            upsert=True  # Create user if doesn't exist
        )
        
        # Remove thread from temp user (optional - can leave it for cleanup later)
        users_collection.update_one(
            {"user_id": request.temp_user_id},
            {
                "$unset": {f"threads.{request.thread_id}": ""}
            }
        )
        
        print(f"✅ Thread {request.thread_id} transferred from {request.temp_user_id} to {request.real_user_id}")
        
        return {
            "success": True,
            "message": f"Thread {request.thread_id} successfully transferred"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transfer thread: {str(e)}")

if __name__ == "__main__":
    # To run this API, use the following command in your terminal:
    # uvicorn endpoints:app --reload --port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)