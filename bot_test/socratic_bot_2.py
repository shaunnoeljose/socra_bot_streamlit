# socratic_bot_logic.py

import os
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, ToolCall
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool # Ensure tool decorator is imported
import json
import uuid # Import uuid for generating unique IDs

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Removed MemorySaver import
# from langgraph.checkpoint.memory import MemorySaver

# Import the logging utility (assuming logger.py will be created later)
# from logger import setup_logger
# logger = setup_logger() # Uncomment when logger.py is ready

# --- Configuration for Memory Management ---
MAX_MESSAGES_IN_CONTEXT = 10 # Keep the last 10 messages in the context window
# This includes both HumanMessage and AIMessage. Adjust as needed based on LLM context limits.

# --- 1. Define the Agent State ---
class SocraticAgentState(TypedDict):
    """
    Represents the state of the Socratic agent's conversation.

    Attributes:
        messages: A list of chat messages exchanged so far.
        difficulty_level: The current difficulty level of questions (e.g., 'beginner', 'intermediate', 'advanced').
        user_struggle_count: Counter for consecutive times the user struggles.
        topic: The current Python topic being discussed.
        sub_topic: The specific sub-topic within the main topic.
        mcq_active: Boolean indicating if an MCQ is currently active.
        mcq_question: The active MCQ question text.
        mcq_options: List of options for the active MCQ.
        mcq_correct_answer: The correct answer for the active MCQ.
        agent_thought: The last thought process articulated by the Socratic agent.
        # Added for supervisor routing
        next_node: str # The next node the supervisor has decided to route to
        tool_input: dict # Input arguments for the tool if a tool is routed to
    """
    messages: Annotated[List[BaseMessage], lambda x, y: x + y] # Appends new messages to the list
    difficulty_level: str
    user_struggle_count: int
    topic: str
    sub_topic: str
    mcq_active: bool
    mcq_question: str
    mcq_options: List[str]
    mcq_correct_answer: str
    agent_thought: str
    next_node: str
    tool_input: dict


# --- 2. Initialize the LLMs and Tools ---

# Initialize the Gemini LLM (used for both Socratic and Supervisor agents)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7) # Changed model to gemini-2.0-flash

# --- Define User-Facing Simulated Agent Tools ---
# These are the actual tools that will perform specific tasks.
@tool
def code_analysis_agent(code: str) -> str:
    """
    Analyzes the provided Python code, identifies potential issues, suggests improvements,
    and provides feedback. Use this when the user provides code and asks for review or debugging.
    The output is raw analysis, which the Socratic agent will then use to ask questions.
    """
    # In a real scenario, this would call another LLM or a static analysis tool.
    # logger.info(f"Executing Code Analysis for: {code[:50]}...") # Uncomment when logger is ready
    return f"Code Analysis Result: For the code snippet '{code}', a potential area to explore is its efficiency in handling large inputs, or error handling. Also, consider adding comments for clarity."

@tool
def code_explanation_agent(concept: str) -> str:
    """
    Explains a given Python concept, function, keyword, or error message in detail.
    Use this when the user asks for an explanation of something.
    The output is raw explanation, which the Socratic agent will then use to ask questions.
    """
    # In a real scenario, this would call another LLM specialized in explanations.
    # logger.info(f"Executing Code Explanation for: {concept}") # Uncomment when logger is ready
    return f"Explanation Result: The concept of '{concept}' in Python generally refers to [brief factual summary]. For instance, if it's about 'loops', it's about repetitive execution. If it's 'objects', it's about data and behavior bundling."

@tool
def challenge_generator_agent(topic: str, difficulty: str) -> str:
    """
    Generates a Python coding challenge or a fill-in-the-blanks exercise based on the specified topic and difficulty.
    Use this when the user requests a challenge.
    The output is the challenge, which the Socratic agent will present.
    """
    # In a real scenario, this would call another LLM or a challenge generation service.
    # logger.info(f"Executing Challenge Generation for: {topic}, Difficulty: {difficulty}") # Uncomment when logger is ready
    return f"Challenge Result: For '{topic}' at '{difficulty}' difficulty: 'Write a Python function that takes a list of numbers and returns the sum of all **odd** numbers.' How would you approach solving this?"

@tool
def mcq_agent(topic: str, difficulty: str) -> str:
    """
    Generates a multiple-choice question (MCQ) on a given Python topic and difficulty level.
    The output will be a JSON string containing the question, options, and correct answer.
    This tool is called when the Socratic agent decides to test understanding via MCQ.
    """
    # In a real scenario, this would call another LLM specifically for MCQ generation.
    # logger.info(f"Executing MCQ Generation for: {topic}, Difficulty: {difficulty}") # Uncomment when logger is ready
    mcq_data = {
        "question": f"Which of the following operations would lead to an `IndentationError` in Python?",
        "options": ["A) Missing a colon after a function definition", "B) Inconsistent use of spaces and tabs for indentation", "C) Using a reserved keyword as a variable name", "D) Forgetting a closing parenthesis"],
        "correct_answer": "B"
    }
    return json.dumps(mcq_data)

# List of all user-facing tools
# Changed to a dictionary for direct lookup by name
user_facing_tools_map = {
    code_analysis_agent.name: code_analysis_agent,
    code_explanation_agent.name: code_explanation_agent,
    challenge_generator_agent.name: challenge_generator_agent,
    mcq_agent.name: mcq_agent,
}


# --- Define Internal Supervisor Tools (for routing decisions) ---
# These are "tools" the supervisor LLM will call to indicate its routing decision.
@tool
def route_to_socratic_question(query: str = None) -> str:
    """Routes the conversation to the main Socratic Questioning agent for general teaching or follow-up.
    This is the default route for general queries, concept discussions, and after tool outputs.
    Optionally includes a follow-up query for the Socratic agent if the intent is specific.
    """
    return "socratic_question"

@tool
def route_to_code_analysis(code: str) -> str:
    """Routes to the Code Analysis agent for debugging or code review. Requires the code snippet."""
    return "code_analysis"

@tool
def route_to_code_explanation(concept: str) -> str:
    """Routes to the Code Explanation agent to explain a specific concept, keyword, or error. Requires the concept."""
    return "code_explanation"

@tool
def route_to_challenge_generator(topic: str = None, difficulty: str = None) -> str:
    """Routes to the Challenge Generator agent to create a coding challenge. Optionally specify topic and difficulty."""
    return "challenge_generator"

@tool
def route_to_mcq_generator(topic: str = None, difficulty: str = None) -> str:
    """Routes to the MCQ Generator agent to create a multiple-choice question. Optionally specify topic and difficulty."""
    return "mcq_generator"

# List of all internal routing tools available to the Supervisor
supervisor_routing_tools = [
    route_to_socratic_question,
    route_to_code_analysis,
    route_to_code_explanation,
    route_to_challenge_generator,
    route_to_mcq_generator
]

# Supervisor Agent Setup
supervisor_system_prompt = """
You are a highly intelligent routing agent for a Socratic Python Tutor. Your task is to analyze
the user's last message and the conversation history to determine the most appropriate next step
in the learning process.

You have access to several internal routing tools. Call exactly one of these tools to specify
which specialized agent or flow should handle the user's request.

Here are your routing rules:
-   **Default:** For general questions, learning new topics, or continuing a Socratic dialogue, use `route_to_socratic_question`. This should be your most frequent choice.
-   **Code Analysis:** If the user provides Python code and asks for debugging, feedback, review, or analysis, use `route_to_code_analysis` and pass the code.
-   **Code Explanation:** If the user explicitly asks for an explanation of a specific Python concept, keyword, function, or error message, use `route_to_code_explanation` and pass the concept.
-   **Challenge/Exercise:** If the user explicitly asks for a coding challenge, exercise, or fill-in-the-blanks, use `route_to_challenge_generator`.
-   **MCQ:** If the user asks for a multiple-choice question or you determine an MCQ is a good way to test their understanding, use `route_to_mcq_generator`.

Pay close attention to keywords and the overall intent. Your response MUST be a tool call.

Current difficulty level: {difficulty_level}
Current topic: {topic}
Current sub-topic: {sub_topic}
"""

# Supervisor Agent Setup
supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", supervisor_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
supervisor_runnable = supervisor_prompt | llm.bind_tools(supervisor_routing_tools)

# Socratic Agent Setup (This is our main Socratic Questioning LLM)
socratic_system_prompt = """
You are a Socratic Python programming tutor. Your goal is to guide the user to discover answers
and understand concepts through thoughtful questions, rather than directly providing solutions.

Here are your core principles:
1.  **Ask Questions:** Always respond with a question, unless explicitly providing feedback on code or an MCQ answer.
2.  **Socratic Method:** Break down complex problems into smaller, manageable questions.
3.  **Encourage Exploration:** Prompt the user to experiment, research, or think critically.
4.  **Adapt to User Understanding:**
    * **Struggle Detection:** If the user seems confused, provides incorrect answers, or asks for direct solutions, simplify your questions, rephrase, or offer a hint.
    * **Progression:** If the user demonstrates understanding, subtly move to a slightly more advanced sub-concept or a related new topic. Avoid repetitive questioning on the same point.
5.  **Interpret Tool Outputs Socratically:** If a tool provides information (e.g., Code Analysis Result, Explanation Result, Challenge Result), your task is to *process that information* and turn it into a Socratic question or guided step for the user. Do not just relay the tool's output directly.
6.  **Maintain Context:** Keep track of the current topic and sub-topic.
7.  **Be Patient and Encouraging:** Foster a positive learning environment.
8.  **Strict Output Format:** Your response MUST be a direct Socratic question. Do NOT include any "Thought:" prefix, internal monologue, conversational filler, or anything other than the question itself. Ensure the question is the ONLY content.

Current difficulty level: {difficulty_level}
Current topic: {topic}
Current sub-topic: {sub_topic}
User struggle count: {user_struggle_count}
MCQ active: {mcq_active}
"""

socratic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", socratic_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
socratic_agent_runnable = socratic_prompt | llm # Socratic agent does not call tools directly, supervisor does

# --- 3. Define the Graph Nodes ---

def call_supervisor(state: SocraticAgentState):
    messages = [
    msg for msg in state["messages"][-MAX_MESSAGES_IN_CONTEXT:]
    if hasattr(msg, "content") and msg.content.strip() != ""
]
    response = supervisor_runnable.invoke({
        "messages": messages,
        "difficulty_level": state["difficulty_level"],
        "user_struggle_count": state["user_struggle_count"],
        "topic": state["topic"],
        "sub_topic": state["sub_topic"]
    })

    tool_name = None
    tool_input = {}

    if response.tool_calls:
        tool_call_item = response.tool_calls[0]

        # Log structure for debugging
        print("Raw tool_call structure:", tool_call_item)

        if isinstance(tool_call_item, dict):
            tool_name = tool_call_item.get("name", None)
            tool_input = tool_call_item.get("args", {})
        elif hasattr(tool_call_item, "function"):
            tool_name = tool_call_item.function.name
            tool_input = tool_call_item.function.arguments
        else:
            print("Warning: Unexpected tool call structure.")
            tool_name = None

    if not tool_name:
        print("Warning: Supervisor LLM returned a malformed tool call (name is None). Defaulting to socratic_question.")
        return {
            "messages": [response],
            "next_node": "socratic_question",
            "tool_input": {}
        }

    next_node = tool_name.replace("route_to_", "")
    return {
        "messages": [response],
        "next_node": next_node,
        "tool_input": tool_input
    }

def socratic_question_node(state: SocraticAgentState):
    """
    Node for the main Socratic LLM to ask questions or interpret tool outputs.
    """
    # logger.info("Socratic Question Node activated.") # Uncomment when logger is ready
    messages = [
    msg for msg in state["messages"][-MAX_MESSAGES_IN_CONTEXT:]
    if hasattr(msg, "content") and msg.content.strip() != ""
]
    response = socratic_agent_runnable.invoke({
        "messages": messages,
        "difficulty_level": state["difficulty_level"],
        "user_struggle_count": state["user_struggle_count"],
        "topic": state["topic"],
        "sub_topic": state["sub_topic"],
        "mcq_active": state["mcq_active"]
    })

    # Removed extraction of agent_thought as per user request
    # thought = ""
    # if response.content and response.content.startswith("Thought:"):
    #     parts = response.content.split("Thought:", 1)
    #     if len(parts) > 1:
    #         thought = parts[1].strip().split('\n', 1)[0]

    # Return without agent_thought
    return {"messages": [response], "agent_thought": ""} # Set agent_thought to empty string


# --- New Nodes for Specialized Tools ---

def code_analysis_node(state: SocraticAgentState):
    """
    Node to execute the code_analysis_agent tool.
    """
    tool_name = "code_analysis_agent"
    tool_args = state["tool_input"]
    tool_function = user_facing_tools_map.get(tool_name)
    tool_output = ""
    if tool_function:
        try:
            # Correct way to invoke the tool function (BaseTool object)
            tool_output = tool_function.invoke(tool_args)
        except Exception as e:
            tool_output = f"Error executing tool {tool_name}: {e}"
    else:
        tool_output = f"Error: Specialized tool '{tool_name}' not found."
    return {"messages": [ToolMessage(content=tool_output, name=tool_name, tool_call_id=str(uuid.uuid4()))]}

def code_explanation_node(state: SocraticAgentState):
    """
    Node to execute the code_explanation_agent tool.
    """
    tool_name = "code_explanation_agent"
    tool_args = state["tool_input"]
    tool_function = user_facing_tools_map.get(tool_name)
    tool_output = ""
    if tool_function:
        try:
            # Correct way to invoke the tool function (BaseTool object)
            tool_output = tool_function.invoke(tool_args)
        except Exception as e:
            tool_output = f"Error executing tool {tool_name}: {e}"
    else:
        tool_output = f"Error: Specialized tool '{tool_name}' not found."
    return {"messages": [ToolMessage(content=tool_output, name=tool_name, tool_call_id=str(uuid.uuid4()))]}

def challenge_generator_node(state: SocraticAgentState):
    """
    Node to execute the challenge_generator_agent tool.
    """
    tool_name = "challenge_generator_agent"
    tool_args = state["tool_input"]
    tool_function = user_facing_tools_map.get(tool_name)
    tool_output = ""
    if tool_function:
        try:
            # Correct way to invoke the tool function (BaseTool object)
            tool_output = tool_function.invoke(tool_args)
        except Exception as e:
            tool_output = f"Error executing tool {tool_name}: {e}"
    else:
        tool_output = f"Error: Specialized tool '{tool_name}' not found."
    return {"messages": [ToolMessage(content=tool_output, name=tool_name, tool_call_id=str(uuid.uuid4()))]}


def generate_mcq_node(state: SocraticAgentState):
    """
    Node specifically for generating an MCQ via the mcq_agent tool.
    This also handles setting the MCQ active state for main.py.
    """
    # logger.info("MCQ Generation Node activated.") # Uncomment when logger is ready
    tool_name = state["next_node"] # This comes from the supervisor's routing decision
    tool_args = state["tool_input"] # Should contain topic and difficulty from supervisor

    # Manually find and execute the tool function using the map
    tool_function = user_facing_tools_map.get(tool_name)
    mcq_raw_output = ""

    if tool_function:
        try:
            # Correct way to invoke the tool function (BaseTool object)
            mcq_raw_output = tool_function.invoke(tool_args)
            mcq_data = json.loads(mcq_raw_output)
            state["mcq_active"] = True
            state["mcq_question"] = mcq_data["question"]
            state["mcq_options"] = mcq_data["options"]
            state["mcq_correct_answer"] = mcq_data["correct_answer"]
            # logger.info("MCQ details updated in state.") # Uncomment when logger is ready
        except Exception as e:
            mcq_raw_output = f"Error generating MCQ: {e}"
            # logger.error(f"Error generating MCQ: {e}", exc_info=True) # Uncomment when logger is ready
    
    # Add a ToolMessage for the MCQ generation, which the Socratic LLM can interpret
    # or simply for logging purposes in the graph flow.
    # Provide a unique tool_call_id for the ToolMessage
    return {"messages": [ToolMessage(content=mcq_raw_output, name=tool_name, tool_call_id=str(uuid.uuid4()))], **state}


# --- 4. Define the Graph Edges (Conditional Logic) ---

def route_supervisor_output(state: SocraticAgentState):
    """
    Conditional edge from the supervisor to determine the next node based on its decision.
    """
    # logger.info(f"Routing supervisor output. Next node: {state['next_node']}") # Uncomment when logger is ready
    if state["next_node"] == "socratic_question":
        return "socratic_question_node"
    elif state["next_node"] == "mcq_generator":
        return "generate_mcq_node"
    # All other specialized tools now have their own nodes
    elif state["next_node"] == "code_analysis":
        return "code_analysis_node"
    elif state["next_node"] == "code_explanation":
        return "code_explanation_node"
    elif state["next_node"] == "challenge_generator":
        return "challenge_generator_node"
    return "socratic_question_node" # Fallback to socratic question if unexpected


# --- 5. Build the LangGraph ---

# Create a StateGraph instance with our defined state.
workflow = StateGraph(SocraticAgentState)

# Add nodes to the workflow.
workflow.add_node("call_supervisor", call_supervisor)
workflow.add_node("socratic_question_node", socratic_question_node) # Renamed from call_llm
# Add new specialized tool nodes
workflow.add_node("code_analysis_node", code_analysis_node)
workflow.add_node("code_explanation_node", code_explanation_node)
workflow.add_node("challenge_generator_node", challenge_generator_node)
workflow.add_node("generate_mcq_node", generate_mcq_node)

# Set the entry point for the graph.
workflow.set_entry_point("call_supervisor")

# Define the edges.
# From supervisor, route conditionally
workflow.add_conditional_edges(
    "call_supervisor",
    route_supervisor_output,
    {
        "socratic_question_node": "socratic_question_node",
        "code_analysis_node": "code_analysis_node",
        "code_explanation_node": "code_explanation_node",
        "challenge_generator_node": "challenge_generator_node",
        "generate_mcq_node": "generate_mcq_node",
    }
)

# After each specialized tool node, return to the socratic_question_node
# for the Socratic LLM to interpret the tool's output and formulate a question.
workflow.add_edge("code_analysis_node", "socratic_question_node")
workflow.add_edge("code_explanation_node", "socratic_question_node")
workflow.add_edge("challenge_generator_node", "socratic_question_node")

# After the socratic_question_node, the run ends. The main.py loop will then take user input.
workflow.add_edge("socratic_question_node", END)

# After generating an MCQ, the run ends. Main.py handles the MCQ display and user input.
workflow.add_edge("generate_mcq_node", END)

# Removed MemorySaver initialization and compilation
# checkpointer = MemorySaver()
socratic_graph = workflow.compile() # Compile without checkpointer

# --- Temporary LLM Connection Test ---
# This function will be called once when the module is imported to test LLM connectivity.
# def _test_llm_connection():
#     print("\n--- Testing LLM Connection (Temporary) ---")
#     try:
#         test_message = HumanMessage(content="Say hi!")
#         response = llm.invoke([test_message])
#         print(f"LLM Test Response: {response.content}")
#         print("LLM connection successful!")
#     except Exception as e:
#         print(f"LLM Connection Test FAILED: {e}")
#         print("Please check your GOOGLE_API_KEY and network connection.")
#     print("--- End LLM Connection Test ---\n")

# # --- Temporary Node Functionality Test ---
# # This function allows testing individual nodes with a sample state.
# def _test_node_functionality(node_name: str, initial_state: SocraticAgentState):
#     print(f"\n--- Testing Node: {node_name} ---")
#     try:
#         node_runnable = workflow.nodes[node_name].runnable
#         result_state_update = node_runnable.invoke(initial_state)

#         print(f"Node '{node_name}' executed successfully.")
#         print(f"Initial State (messages only): {[msg.content if hasattr(msg, 'content') else str(msg) for msg in initial_state['messages']]}")
#         print(f"Resulting State Update: {result_state_update}")

#         if 'messages' in result_state_update and result_state_update['messages']:
#             print("Messages returned by node:")
#             for msg in result_state_update['messages']:
#                 if isinstance(msg, BaseMessage):
#                     print(f"  - Type: {msg.type}, Content: {msg.content[:50]}...")
#                     if hasattr(msg, 'tool_calls') and msg.tool_calls:
#                         for tc in msg.tool_calls:
#                             if hasattr(tc, "function"):
#                                 tool_name_debug = tc.function.name
#                                 tool_args_debug = tc.function.arguments
#                             elif isinstance(tc, dict):
#                                 tool_name_debug = tc.get("name")
#                                 tool_args_debug = tc.get("args")
#                             else:
#                                 tool_name_debug = "unknown"
#                                 tool_args_debug = {}
#                             print(f"    Tool Call: {tool_name_debug} with args {tool_args_debug}")
#                 else:
#                     print(f"  - Raw: {msg}")

#     except Exception as e:
#         print(f"Node '{node_name}' Test FAILED: {e}")
#         import traceback
#         traceback.print_exc()
#     print(f"--- End Node Test: {node_name} ---\n")


# # Call the LLM connection test immediately when this module is loaded
# _test_llm_connection()

# # Example usage of the new node testing function (uncomment to test specific nodes)
# # Ensure your GOOGLE_API_KEY is set for LLM-based nodes.

# # Test call_supervisor node
# sample_supervisor_state = SocraticAgentState(
#     messages=[HumanMessage(content="I need help with debugging this code: print('hello')")],
#     difficulty_level="beginner",
#     user_struggle_count=0,
#     topic="Python Basics",
#     sub_topic="Introduction",
#     mcq_active=False,
#     mcq_question="", mcq_options=[], mcq_correct_answer="",
#     agent_thought="", next_node="", tool_input={}
# )
# _test_node_functionality("call_supervisor", sample_supervisor_state)

# # Test socratic_question_node
# sample_socratic_state = SocraticAgentState(
#     messages=[HumanMessage(content="What are variables?")], # Removed the AIMessage from here
#     difficulty_level="beginner",
#     user_struggle_count=0,
#     topic="Variables",
#     sub_topic="Definition",
#     mcq_active=False,
#     mcq_question="", mcq_options=[], mcq_correct_answer="",
#     agent_thought="", next_node="", tool_input={}
# )
# _test_node_functionality("socratic_question_node", sample_socratic_state)

# # Test code_analysis_node
# sample_tool_state_code_analysis = SocraticAgentState(
#     messages=[HumanMessage(content="debug this code: print('hello')")],
#     difficulty_level="beginner",
#     user_struggle_count=0,
#     topic="Debugging",
#     sub_topic="Code Analysis",
#     mcq_active=False,
#     mcq_question="", mcq_options=[], mcq_correct_answer="",
#     agent_thought="",
#     next_node="code_analysis_agent", # This should match the tool's actual name
#     tool_input={"code": "def my_func():\n    pass"}
# )
# _test_node_functionality("code_analysis_node", sample_tool_state_code_analysis)

# # Test code_explanation_node
# sample_tool_state_code_explanation = SocraticAgentState(
#     messages=[HumanMessage(content="Explain loops in Python")],
#     difficulty_level="beginner",
#     user_struggle_count=0,
#     topic="Python Basics",
#     sub_topic="Loops",
#     mcq_active=False,
#     mcq_question="", mcq_options=[], mcq_correct_answer="",
#     agent_thought="",
#     next_node="code_explanation_agent",
#     tool_input={"concept": "loops"}
# )
# _test_node_functionality("code_explanation_node", sample_tool_state_code_explanation)

# # Test challenge_generator_node
# sample_tool_state_challenge_generator = SocraticAgentState(
#     messages=[HumanMessage(content="Give me a challenge on functions.")],
#     difficulty_level="intermediate",
#     user_struggle_count=0,
#     topic="Functions",
#     sub_topic="Challenges",
#     mcq_active=False,
#     mcq_question="", mcq_options=[], mcq_correct_answer="",
#     agent_thought="",
#     next_node="challenge_generator_agent",
#     tool_input={"topic": "functions", "difficulty": "intermediate"}
# )
# _test_node_functionality("challenge_generator_node", sample_tool_state_challenge_generator)

# # Test generate_mcq_node
# sample_mcq_state = SocraticAgentState(
#     messages=[HumanMessage(content="Give me an MCQ on functions.")],
#     difficulty_level="beginner",
#     user_struggle_count=0,
#     topic="Functions",
#     sub_topic="MCQ",
#     mcq_active=False,
#     mcq_question="", mcq_options=[], mcq_correct_answer="",
#     agent_thought="",
#     next_node="mcq_agent", # Supervisor would set this
#     tool_input={"topic": "functions", "difficulty": "beginner"} # Supervisor would set this
# )
# _test_node_functionality("generate_mcq_node", sample_mcq_state)
