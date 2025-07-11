# socratic_bot_logic_enhanced.py
import os
from typing import List, TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import json
from langchain_groq import ChatGroq
import uuid # Import uuid for generating unique IDs

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# --- 1. Define the Agent State ---
class SocraticAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    difficulty_level: str
    user_struggle_count: int
    topic: str
    sub_topic: str
    mcq_active: bool
    mcq_question: str
    mcq_options: List[str]
    mcq_correct_answer: str
    agent_thought: str
    interaction_mode: str  # 'general', 'code_review', 'concept_exploration', 'challenge', 'mcq', 'mcq_request'
    context_data: dict  # Store relevant context like code, concept, etc.

# --- 2. Initialize the LLMs ---
groq_api_key = os.getenv("GROQ_API")
# Ensure GROQ_API is set in your .env file for ChatGroq to work
if not groq_api_key:
    # Fallback to Google Generative AI if GROQ_API is not set
    print("GROQ_API not found, falling back to ChatGoogleGenerativeAI (gemini-2.5-flash).")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    mcq_generation_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    supervisor_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
else:
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.7, api_key=groq_api_key)
    mcq_generation_llm = ChatGroq(model="llama3-70b-8192", temperature=0.5, api_key=groq_api_key)
    supervisor_llm = ChatGroq(model="llama3-70b-8192", temperature=0.3, api_key=groq_api_key)


# --- 3. Socratic Tools (Information Gathering Only) ---
@tool
def extract_code_context(code: str) -> str:
    """
    Extracts key information about the provided code for Socratic questioning.
    This tool analyzes code structure, patterns, and potential issues to inform questions.
    """
    context = {
        "code_length": len(code.split('\n')),
        "has_functions": "def " in code,
        "has_classes": "class " in code,
        "has_loops": any(loop in code for loop in ["for ", "while "]),
        "has_conditionals": any(cond in code for cond in ["if ", "elif ", "else"]),
        "imports": [line.strip() for line in code.split('\n') if line.strip().startswith('import') or line.strip().startswith('from')],
        "potential_issues": []
    }
    
    # Check for common issues
    if code.count('(') != code.count(')'):
        context["potential_issues"].append("parentheses_mismatch")
    if "print(" in code and code.count("print(") > 3:
        context["potential_issues"].append("excessive_prints")
    if "global " in code:
        context["potential_issues"].append("global_variables")
    
    return json.dumps(context)

@tool
def analyze_concept_depth(concept: str) -> str:
    """
    Analyzes the depth and complexity of a programming concept, including its direct relevance
    to Python and suggesting Python alternatives if applicable.
    Returns JSON with 'level', 'prerequisites', 'subtopics', 'python_relevance', 'python_alternative'.
    python_relevance: 'core', 'related_but_different', 'not_directly_python'
    """
    concept_mapping = {
        "variables": {"level": "beginner", "prerequisites": [], "subtopics": ["assignment", "naming", "types", "scope"], "python_relevance": "core", "python_alternative": None},
        "functions": {"level": "beginner", "prerequisites": ["variables"], "subtopics": ["definition", "parameters", "return", "scope"], "python_relevance": "core", "python_alternative": None},
        "classes": {"level": "intermediate", "prerequisites": ["functions", "variables"], "subtopics": ["attributes", "methods", "inheritance", "encapsulation"], "python_relevance": "core", "python_alternative": None},
        "loops": {"level": "beginner", "prerequisites": ["variables", "conditionals"], "subtopics": ["for", "while", "iteration", "break", "continue"], "python_relevance": "core", "python_alternative": None},
        "conditionals": {"level": "beginner", "prerequisites": ["variables", "comparisons"], "subtopics": ["if", "elif", "else", "boolean", "logical_operators"], "python_relevance": "core", "python_alternative": None},
        "decorators": {"level": "advanced", "prerequisites": ["functions", "closures"], "subtopics": ["syntax", "parameters", "multiple", "built_in"], "python_relevance": "core", "python_alternative": None},
        "generators": {"level": "advanced", "prerequisites": ["functions", "loops"], "subtopics": ["yield", "iterator", "memory", "lazy_evaluation"], "python_relevance": "core", "python_alternative": None},
        "array": {"level": "beginner", "prerequisites": ["lists"], "subtopics": ["definition", "indexing", "mutability", "numpy_arrays"], "python_relevance": "related_but_different", "python_alternative": "list"},
        "list": {"level": "beginner", "prerequisites": ["variables"], "subtopics": ["creation", "indexing", "slicing", "methods", "mutability"], "python_relevance": "core", "python_alternative": None},
        "pointer": {"level": "advanced", "prerequisites": [], "subtopics": [], "python_relevance": "not_directly_python", "python_alternative": "references or object identity"},
        "struct": {"level": "intermediate", "prerequisites": [], "subtopics": [], "python_relevance": "not_directly_python", "python_alternative": "dictionaries or classes"}
    }
    
    concept_info = concept_mapping.get(concept.lower(), {
        "level": "intermediate", 
        "prerequisites": ["basic_python"], 
        "subtopics": ["definition", "usage", "examples"],
        "python_relevance": "not_directly_python", # Default to not directly Python if not explicitly defined
        "python_alternative": "a related Python concept (e.g., for 'array', consider 'list')" # Generic fallback
    })
    
    return json.dumps(concept_info)

@tool
def generate_mcq_data(topic: str, difficulty: str) -> str:
    """
    Generates MCQ data for Socratic assessment. First tries predefined, then generates new ones.
    """
    mcqs_raw = {
        "variables": {
            "question": "Which of the following data types is mutable in Python?",
            "options": ["A) Tuple", "B) String", "C) List", "D) Integer"],
            "correct_answer": "C"
        },
        "functions": {
            "question": "Which keyword is used to define a function in Python?",
            "options": ["A) func", "B) define", "C) def", "D) function"],
            "correct_answer": "C"
        },
        "classes": {
            "question": "In Python, what is the primary purpose of the `__init__` method in a class?",
            "options": [
                "A) To destroy an object when it's no longer needed",
                "B) To define static methods",
                "C) To initialize the attributes of an object when it's created",
                "D) To define the string representation of an object"
            ],
            "correct_answer": "C"
        },
        "loops": {
            "question": "What will happen if you don't include a break statement in a while loop with a condition that never becomes False?",
            "options": ["A) The program will end normally", "B) An error will occur", "C) The loop will run indefinitely", "D) Python will automatically break the loop"],
            "correct_answer": "C"
        },
        "conditionals": {
            "question": "Which Python keyword is used to start an 'if' statement?",
            "options": ["A) then", "B) if", "C) when", "D) check"],
            "correct_answer": "B"
        },
        "array": { # Added predefined MCQ for 'array'
            "question": "Which Python built-in data structure is most similar to a traditional array and allows mutable, ordered collections of items?",
            "options": ["A) Tuple", "B) Dictionary", "C) List", "D) Set"],
            "correct_answer": "C"
        },
        "list": { # Added predefined MCQ for 'list'
            "question": "Which of the following is true about Python lists?",
            "options": ["A) They are immutable", "B) They can store only one data type", "C) They are ordered and mutable", "D) They are defined using curly braces {}"],
            "correct_answer": "C"
        }
    }
    
    selected_mcq = mcqs_raw.get(topic.lower())
    
    if selected_mcq:
        formatted_question = f"**{selected_mcq['question']}**\n\n" + "\n".join(selected_mcq['options'])
        return json.dumps({
            "question": formatted_question,
            "options": selected_mcq['options'],
            "correct_answer": selected_mcq['correct_answer']
        })
    else:
        # Generate using LLM
        mcq_prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a Python MCQ in JSON format with 'question', 'options' (array of 4 strings A-D), and 'correct_answer' (A/B/C/D).
            Ensure the JSON is perfectly valid and contains all three keys: 'question', 'options', and 'correct_answer'.
            The question should be about Python programming concepts.
            Example: {"question": "What is 2+2?", "options": ["A) 3", "B) 4", "C) 5", "D) 6"], "correct_answer": "B"}
            """),
            ("user", f"Topic: {topic}, Difficulty: {difficulty}")
        ])
        
        try:
            response = mcq_generation_llm.invoke(mcq_prompt.format())
            mcq_data = json.loads(response.content)
            
            # Basic validation of the generated MCQ structure
            if all(k in mcq_data for k in ["question", "options", "correct_answer"]) and \
               isinstance(mcq_data["options"], list) and len(mcq_data["options"]) == 4 and \
               mcq_data["correct_answer"] in ["A", "B", "C", "D"]:
                
                formatted_question = f"**{mcq_data['question']}**\n\n" + "\n".join(mcq_data['options'])
                return json.dumps({
                    "question": formatted_question,
                    "options": mcq_data['options'],
                    "correct_answer": mcq_data['correct_answer']
                })
            else:
                # Fallback if generated JSON is invalid
                print(f"Invalid MCQ structure from LLM: {mcq_data}") # For debugging
                return json.dumps({"error": "Generated MCQ has invalid structure."})
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e} - Response content: {response.content}") # For debugging
            return json.dumps({"error": "Could not parse generated MCQ (invalid JSON)."})
        except Exception as e:
            print(f"General Error generating MCQ: {e}") # For debugging
            return json.dumps({"error": f"An unexpected error occurred during MCQ generation: {str(e)}"})

@tool
def create_challenge_context(topic: str, difficulty: str) -> str:
    """
    Creates challenge context for Socratic guidance through problem-solving.
    """
    challenges = {
        "variables": {
            "problem": "Swapping two variables without using a third variable",
            "key_concepts": ["assignment", "arithmetic", "temporary storage"],
            "guiding_questions": [
                "What methods do you know for swapping values?",
                "How might mathematical operations help?",
                "What happens when you do a = a + b?"
            ]
        },
        "functions": {
            "problem": "Calculating factorial of a number",
            "key_concepts": ["recursion", "iteration", "base case"],
            "guiding_questions": [
                "What is a factorial mathematically?",
                "How would you break this down into smaller problems?",
                "What happens when the number is 0 or 1?"
            ]
        },
        "loops": {
            "problem": "Generating Fibonacci sequence",
            "key_concepts": ["iteration", "sequence", "previous values"],
            "guiding_questions": [
                "How does each Fibonacci number relate to previous ones?",
                "What values do you need to track?",
                "How would you generate the next number?"
            ]
        },
        "classes": {
            "problem": "Creating a simple bank account class",
            "key_concepts": ["encapsulation", "methods", "attributes"],
            "guiding_questions": [
                "What data should a bank account store?",
                "What operations can you perform on an account?",
                "How would you ensure the balance can't be negative?"
            ]
        }
    }
    
    return json.dumps(challenges.get(topic.lower(), {
        "problem": f"Implementing a solution for {topic}",
        "key_concepts": ["problem_solving", "implementation"],
        "guiding_questions": [f"How would you approach solving {topic}?"]
    }))

# --- 4. Supervisor Node ---
supervisor_system_prompt = """
You are a supervisor for a Socratic Python tutoring system. Your role is to determine the interaction mode 
based on the user's message and current state. Remember: EVERYTHING must remain Socratic - we never give direct answers.

Available interaction modes:
1. "general" - General Socratic questioning and topic exploration
2. "code_review" - Socratic code review through guided questions
3. "concept_exploration" - Deep dive into concepts through questioning
4. "challenge" - Guiding through problem-solving via questions
5. "mcq_active" - When user is answering an MCQ
6. "mcq_request" - When user wants an MCQ or needs assessment
7. "evaluate_understanding" - When the user gives a short, affirmative response (e.g., "Got it", "Ok", "Yes") and the tutor needs to determine if it signifies true understanding or requires further probing.

Current state:
- Difficulty: {difficulty_level}
- Topic: {topic}
- Sub-topic: {sub_topic}
- Struggle count: {user_struggle_count}
- MCQ active: {mcq_active}
- Current mode: {interaction_mode}

User message indicators:
- Code blocks/snippets → "code_review"
- "explain", "what is", "how does" → "concept_exploration"  
- "challenge", "problem", "exercise" → "challenge"
- "quiz", "test", "MCQ" → "mcq_request"
- A/B/C/D answers when MCQ active → "mcq_active"
- Short affirmative responses like "Got it", "Ok", "Yes", "I understand" → "evaluate_understanding"
- General questions or detailed responses → "general"

Respond with just the mode name.
"""

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", supervisor_system_prompt),
    ("user", "User message: {user_message}")
])

def supervisor_node(state: SocraticAgentState):
    """Determines the interaction mode for Socratic guidance."""
    last_message = state["messages"][-1]
    user_message = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # Quick checks for specific patterns
    if state.get("mcq_active") and user_message.strip().upper() in ["A", "B", "C", "D"]:
        return {"interaction_mode": "mcq_active"}
    
    if any(keyword in user_message.lower() for keyword in ["```", "def ", "class ", "import ", "print("]):
        return {"interaction_mode": "code_review", "context_data": {"code": user_message}}
    
    # Check for short affirmative responses to trigger 'evaluate_understanding'
    short_affirmative_keywords = ["got it", "ok", "yes", "i understand", "understood", "ahh i see", "i get it now"]
    if any(keyword in user_message.lower() for keyword in short_affirmative_keywords) and len(user_message.split()) <= 5:
        # If it's a short affirmative, force evaluation
        return {"interaction_mode": "evaluate_understanding"}

    # Use supervisor LLM for more complex decisions
    response = supervisor_llm.invoke(supervisor_prompt.format(
        user_message=user_message,
        difficulty_level=state.get("difficulty_level", "beginner"),
        topic=state.get("topic", ""),
        sub_topic=state.get("sub_topic", ""),
        user_struggle_count=state.get("user_struggle_count", 0),
        mcq_active=state.get("mcq_active", False),
        interaction_mode=state.get("interaction_mode", "general")
    ))
    
    mode = response.content.strip().lower()
    
    # Extract context based on mode
    context_data = {}
    if mode == "code_review":
        context_data = {"code": user_message}
    elif mode == "concept_exploration" or mode == "mcq_request": # Also extract concept for mcq_request
        # Extract concept from message
        concept_keywords = ["variables", "functions", "classes", "loops", "conditionals", "decorators", "generators", "array", "list", "pointer", "struct"] # Added more keywords
        found_concept = None
        for keyword in concept_keywords:
            if keyword in user_message.lower():
                found_concept = keyword
                break
        if found_concept:
            context_data = {"concept": found_concept}
        else:
            context_data = {"concept": user_message} # Fallback to user message if no specific concept found
    elif mode == "challenge":
        context_data = {"topic": state.get("topic", "general")}
    
    return {"interaction_mode": mode, "context_data": context_data}

# --- 5. Unified Socratic Node ---
socratic_system_prompt = """
You are a Socratic Python programming tutor. Your CORE PRINCIPLE is to NEVER give direct answers. 
Instead, guide students to discover answers through thoughtful questioning.

INTERACTION MODES:
1. GENERAL: Explore topics through open-ended questions
2. CODE_REVIEW: Guide code improvement through questions about structure, logic, and best practices
3. CONCEPT_EXPLORATION: Deep dive into concepts by asking about understanding, applications, and connections
4. CHALLENGE: Guiding through problem-solving via questions
5. MCQ_ACTIVE: Process MCQ answers and provide Socratic feedback
6. MCQ_REQUEST: The user has requested an MCQ.
    First, determine if the requested topic is a core Python concept or if there's a more appropriate Python alternative.
    To do this, you MUST call the `analyze_concept_depth` tool with the requested topic.
    Based on the tool's output (available in context_data.concept_info after tool execution):
    - If `context_data.concept_info.python_relevance` is 'not_directly_python' or 'related_but_different':
        Thought: The user requested an MCQ on a non-Python concept. I will inform them and suggest an alternative.
        In Python, we don't typically use '{requested_topic}' in the same way some other languages do. We often use '{python_alternative}' instead. Would you like an MCQ on Python '{python_alternative}' to test your understanding of that similar concept?
    - If `context_data.concept_info.python_relevance` is 'core':
        Thought: The user requested an MCQ on a core Python concept. I will proceed to generate the MCQ.
        Now, let's test your knowledge!
        <tool_code>
        print(generate_mcq_data(topic='{requested_topic}', difficulty='{difficulty_level}'))
        </tool_code>
    - If `context_data.concept_info` is not yet available (meaning `analyze_concept_depth` hasn't run or failed):
        Thought: I need to analyze the concept depth before generating an MCQ.
        <tool_code>
        print(analyze_concept_depth(concept='{requested_topic}'))
        </tool_code>
    - Always maintain a Socratic tone.
7. EVALUATE_UNDERSTANDING: The user has given a short affirmative response (e.g., "Got it"). Your task is to ask a probing, Socratic question to verify their understanding. Do NOT simply acknowledge their "Got it". Instead, ask them to elaborate, apply the concept, or explain it in their own words. If user_struggle_count is high, consider rephrasing or simplifying.

SOCRATIC PRINCIPLES:
- Always ask questions, never state facts directly
- Build on student's existing knowledge
- Guide discovery through smaller questions
- Encourage experimentation and thinking
- Adapt question complexity to student's understanding
- Use "What do you think...?", "How might...?", "Can you explain...?" patterns

Current Context:
- Mode: {interaction_mode}
- Topic: {topic}
- Difficulty: {difficulty_level}
- Struggle count: {user_struggle_count}
- MCQ active: {mcq_active}
- Context data: {context_data}

RESPONSE FORMAT:
Always start with "Thought: [your reasoning]" then provide your Socratic question or response.
If you need tool information, call the appropriate tool first, then ask questions based on the results.
"""

socratic_prompt = ChatPromptTemplate.from_messages([
    ("system", socratic_system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

# Tools available to the Socratic agent
tools = [extract_code_context, analyze_concept_depth, generate_mcq_data, create_challenge_context]
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")
socratic_agent_runnable = socratic_prompt | llm_with_tools

def socratic_agent_node(state: SocraticAgentState):
    """
    Unified Socratic agent that handles all interactions through questioning.
    """
    # Dynamically set requested_topic for the prompt
    requested_topic = state.get("context_data", {}).get("concept", state.get("topic", "programming"))
    
    # Prepare messages for the LLM, excluding the system prompt part that contains tool calls
    # as we will handle tool calling logic explicitly based on interaction_mode
    messages_for_llm = state["messages"]

    # If it's an MCQ_REQUEST, we need to guide the LLM's thought process
    if state.get("interaction_mode") == "mcq_request":
        concept_info = state.get("context_data", {}).get("concept_info")
        
        if concept_info:
            python_relevance = concept_info.get("python_relevance")
            python_alternative = concept_info.get("python_alternative")

            if python_relevance in ['not_directly_python', 'related_but_different']:
                # Generate conversational response
                response_content = (
                    f"Thought: The user requested an MCQ on a non-Python concept. "
                    f"I will inform them and suggest an alternative.\n"
                    f"In Python, we don't typically use '{requested_topic}' in the same way some other languages do. "
                    f"We often use '{python_alternative}' instead. Would you like an MCQ on Python '{python_alternative}' "
                    f"to test your understanding of that similar concept?"
                )
                return {
                    "messages": [AIMessage(content=response_content)],
                    "agent_thought": "Informed user about non-Python concept and suggested alternative for MCQ."
                }
            elif python_relevance == 'core':
                # Call generate_mcq_data tool
                response_content = (
                    f"Thought: The user requested an MCQ on a core Python concept. "
                    f"I will proceed to generate the MCQ.\n"
                    f"Now, let's test your knowledge!"
                )
                tool_call = {
                    "name": "generate_mcq_data",
                    "args": {"topic": requested_topic, "difficulty": state.get("difficulty_level", "beginner")},
                    "id": str(uuid.uuid4()) # Add a unique ID
                }
                return {
                    "messages": [AIMessage(content=response_content, tool_calls=[tool_call])],
                    "agent_thought": "Called generate_mcq_data tool for core Python concept."
                }
            else:
                # Fallback if concept_info is incomplete or unexpected
                response_content = (
                    f"Thought: Concept info is incomplete or unexpected. Asking for clarification or trying general mode.\n"
                    f"I'm having a bit of trouble with that concept. Can you tell me more about what you'd like to learn or test about '{requested_topic}'?"
                )
                return {
                    "messages": [AIMessage(content=response_content)],
                    "agent_thought": "Concept info incomplete, asking for clarification."
                }
        else:
            # If concept_info is not yet available, call analyze_concept_depth
            response_content = (
                f"Thought: I need to analyze the concept depth before generating an MCQ.\n"
                f"Let me quickly check the relevance of '{requested_topic}' in Python."
            )
            tool_call = {
                "name": "analyze_concept_depth",
                "args": {"concept": requested_topic},
                "id": str(uuid.uuid4()) # Add a unique ID
            }
            return {
                "messages": [AIMessage(content=response_content, tool_calls=[tool_call])],
                "agent_thought": "Called analyze_concept_depth tool."
            }
    
    # For other interaction modes, or if MCQ_REQUEST logic above didn't return
    # This part uses the LLM to generate a response based on the general prompt
    # We need to ensure that requested_topic and python_alternative are always available
    # for the prompt if the interaction mode is not mcq_request but the prompt
    # still contains the conditional logic for these variables.
    
    # Extract these from context_data if they exist, otherwise default to None
    current_context_data = state.get("context_data", {})
    prompt_requested_topic = current_context_data.get("concept", state.get("topic", "programming"))
    prompt_python_alternative = current_context_data.get("concept_info", {}).get("python_alternative")

    response = socratic_agent_runnable.invoke({
        "messages": messages_for_llm, 
        "interaction_mode": state.get("interaction_mode", "general"),
        "topic": state.get("topic", ""),
        "difficulty_level": state.get("difficulty_level", "beginner"),
        "user_struggle_count": state.get("user_struggle_count", 0),
        "mcq_active": state.get("mcq_active", False),
        "context_data": current_context_data, # Pass current context_data
        "requested_topic": prompt_requested_topic, # Explicitly pass requested_topic
        "python_alternative": prompt_python_alternative # Explicitly pass python_alternative
    })
    
    # Extract thought and content more robustly
    thought = ""
    actual_content = response.content # Store the original content
    display_content = ""

    if actual_content:
        if actual_content.startswith("Thought:"):
            parts = actual_content.split("Thought:", 1)
            if len(parts) > 1:
                thought_and_rest = parts[1].strip()
                thought_lines = thought_and_rest.split('\n', 1)
                thought = thought_lines[0]
                
                if len(thought_lines) > 1:
                    display_content = thought_lines[1].strip()
                else:
                    display_content = thought_and_rest 
            else:
                display_content = actual_content
        else:
            display_content = actual_content
    
    # Handle tool calls if any
    if response.tool_calls:
        return {
            "messages": [AIMessage(content=display_content, tool_calls=response.tool_calls)],
            "agent_thought": thought
        }
    else:
        return {
            "messages": [AIMessage(content=display_content)],
            "agent_thought": thought
        }

# --- 6. Tool Execution Node ---
def execute_tools(state: SocraticAgentState):
    """
    Executes tools and continues with Socratic questioning based on results.
    """
    last_message = state["messages"][-1]
    
    if not (isinstance(last_message, AIMessage) and last_message.tool_calls):
        return {"messages": []}
    
    tool_mapping = {
        "extract_code_context": extract_code_context,
        "analyze_concept_depth": analyze_concept_depth,
        "generate_mcq_data": generate_mcq_data,
        "create_challenge_context": create_challenge_context
    }
    
    tool_messages = []
    state_updates = {}
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name in tool_mapping:
            try:
                result = tool_mapping[tool_name].invoke(tool_args)
                tool_messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
                
                # Handle specific tool results
                if tool_name == "generate_mcq_data":
                    try:
                        mcq_data = json.loads(result)
                        if "error" not in mcq_data:
                            state_updates.update({
                                "mcq_active": True,
                                "mcq_question": mcq_data.get("question", ""),
                                "mcq_options": mcq_data.get("options", []),
                                "mcq_correct_answer": mcq_data.get("correct_answer", "")
                            })
                        else:
                            # If MCQ generation failed, set mcq_active to False and provide a message
                            error_message = "Sorry, I couldn't generate an MCQ on that topic right now. Can we try another topic or a different type of interaction?"
                            state_updates.update({
                                "mcq_active": False,
                                "mcq_question": "", # Clear previous MCQ data
                                "mcq_options": [],
                                "mcq_correct_answer": "",
                                "interaction_mode": "general" # Revert to general mode
                            })
                            tool_messages.append(AIMessage(content=f"Thought: Failed to generate MCQ. Reverting to general mode.\n{error_message}"))
                    except json.JSONDecodeError as e:
                        error_message = f"Sorry, I encountered an issue parsing the MCQ. Can we try another topic or a different type of interaction?"
                        print(f"JSON Decode Error in execute_tools for generate_mcq_data: {e} - Raw result: {result}")
                        state_updates.update({
                            "mcq_active": False,
                            "mcq_question": "", 
                            "mcq_options": [],
                            "mcq_correct_answer": "",
                            "interaction_mode": "general"
                        })
                        tool_messages.append(AIMessage(content=f"Thought: JSON parsing failed for generated MCQ. Reverting to general mode.\n{error_message}"))
                elif tool_name == "analyze_concept_depth":
                    try:
                        concept_info = json.loads(result)
                        # Store concept info for socratic_agent to use
                        state_updates["context_data"] = {**state.get("context_data", {}), "concept_info": concept_info}
                        # Also set the topic in the main state if it's a new concept being analyzed
                        if concept_info.get("python_relevance") == "core" or \
                           concept_info.get("python_relevance") == "related_but_different":
                            state_updates["topic"] = tool_args.get("concept", state.get("topic"))
                            state_updates["sub_topic"] = concept_info.get("subtopics", ["Introduction"])[0]
                    except json.JSONDecodeError as e:
                        print(f"JSON Decode Error in execute_tools for analyze_concept_depth: {e} - Raw result: {result}")
                        tool_messages.append(AIMessage(content=f"Thought: Failed to analyze concept depth due to parsing error.\nI'm having a bit of trouble understanding that concept. Can you try rephrasing or asking about a different topic?"))
                        state_updates["interaction_mode"] = "general" # Revert to general
                        
            except Exception as e:
                tool_messages.append(ToolMessage(content=f"Error executing tool '{tool_name}': {str(e)}", tool_call_id=tool_call["id"]))
    
    return {"messages": tool_messages, **state_updates}

# --- 7. MCQ Processing Node ---
def process_mcq_answer(state: SocraticAgentState):
    """
    Processes MCQ answers in a Socratic manner.
    If the answer is incorrect, it increments struggle count and routes back to supervisor.
    """
    last_message = state["messages"][-1]
    user_answer = last_message.content.strip().upper()
    correct_answer = state.get("mcq_correct_answer", "")
    
    # Extract just the letter from the user's response, e.g., "A)" -> "A"
    if len(user_answer) > 0 and user_answer[0] in ["A", "B", "C", "D"]:
        user_answer_letter = user_answer[0]
    else:
        user_answer_letter = "" # Invalid answer format

    is_correct = user_answer_letter == correct_answer
    
    new_struggle_count = state.get("user_struggle_count", 0)
    feedback_message = ""
    
    if is_correct:
        feedback_message = "Excellent! You got it right! "
        new_struggle_count = 0 # Reset struggle count on correct answer
    else:
        feedback_message = f"I see you chose {user_answer_letter}. That's an interesting choice! "
        new_struggle_count += 1
    
    # Add a Socratic follow-up question directly here
    feedback_message += "Now, can you explain your reasoning? What made you choose this answer, and how does it relate to the concept?"

    return {
        "messages": [AIMessage(content=feedback_message)],
        "user_struggle_count": new_struggle_count,
        "mcq_active": False, # Always set MCQ to inactive after processing an answer
        "mcq_question": "",
        "mcq_options": [],
        "mcq_correct_answer": "",
        "interaction_mode": "general", # Set to general, supervisor will re-evaluate
    }

# --- 8. Graph Construction ---
def should_continue_to_tools(state: SocraticAgentState):
    """Check if we need to execute tools."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "execute_tools"
    return "end"

def route_from_supervisor(state: SocraticAgentState):
    """Route based on interaction mode."""
    mode = state.get("interaction_mode", "general")
    if mode == "mcq_active":
        return "process_mcq"
    elif mode == "evaluate_understanding":
        return "socratic_agent"
    return "socratic_agent"

def build_enhanced_socratic_graph():
    """Build the graph with supervisor and unified Socratic approach."""
    workflow = StateGraph(SocraticAgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("socratic_agent", socratic_agent_node)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("process_mcq", process_mcq_answer)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add edges
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "socratic_agent": "socratic_agent",
            "process_mcq": "process_mcq",
        }
    )
    
    workflow.add_conditional_edges(
        "socratic_agent",
        should_continue_to_tools,
        {
            "execute_tools": "execute_tools",
            "end": END
        }
    )
    
    workflow.add_edge("execute_tools", "socratic_agent")
    
    # Modified edge: process_mcq now routes back to supervisor
    workflow.add_edge("process_mcq", "supervisor") 
    
    # Add memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# --- 9. Initialize System ---
enhanced_socratic_graph = build_enhanced_socratic_graph()
# The memory_saver instance needs to be accessible from main.py
memory_saver = MemorySaver() # Define memory_saver here for import in main.py
