import streamlit as st
import os
import json
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, ToolCall
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
import uuid

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration for Memory Management ---
MAX_MESSAGES_IN_CONTEXT = 10

# --- 1. Define the Agent State ---
class SocraticAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
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
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

@tool
def code_analysis_agent(code: str) -> str:
    """
    Analyzes the provided Python code, identifies potential issues, suggests improvements,
    and provides feedback. Use this when the user provides code and asks for review or debugging.
    The output is raw analysis, which the Socratic agent will then use to ask questions.
    """
    return f"Code Analysis Result: For the code snippet '{code}', a potential area to explore is its efficiency in handling large inputs, or error handling. Also, consider adding comments for clarity."

@tool
def code_explanation_agent(concept: str) -> str:
    """
    Explains a given Python concept, function, keyword, or error message in detail.
    Use this when the user asks for an explanation of something.
    The output is raw explanation, which the Socratic agent will then use to ask questions.
    """
    return f"Explanation Result: The concept of '{concept}' in Python generally refers to [brief factual summary]. For instance, if it's about 'loops', it's about repetitive execution. If it's 'objects', it's about data and behavior bundling."

@tool
def challenge_generator_agent(topic: str, difficulty: str) -> str:
    """
    Generates a Python coding challenge or a fill-in-the-blanks exercise based on the specified topic and difficulty.
    Use this when the user requests a challenge.
    The output is the challenge, which the Socratic agent will present.
    """
    return f"Challenge Result: For '{topic}' at '{difficulty}' difficulty: 'Write a Python function that takes a list of numbers and returns the sum of all **odd** numbers.' How would you approach solving this?"

@tool
def mcq_agent(topic: str, difficulty: str) -> str:
    """
    Generates a multiple-choice question (MCQ) on a given Python topic and difficulty level.
    The output will be a JSON string containing the question, options, and correct answer.
    This tool is called when the Socratic agent decides to test understanding via MCQ.
    """
    mcq_data = {
        "question": f"Which of the following operations would lead to an `IndentationError` in Python?",
        "options": ["A) Missing a colon after a function definition", "B) Inconsistent use of spaces and tabs for indentation", "C) Using a reserved keyword as a variable name", "D) Forgetting a closing parenthesis"],
        "correct_answer": "B"
    }
    return json.dumps(mcq_data)

user_facing_tools_map = {
    code_analysis_agent.name: code_analysis_agent,
    code_explanation_agent.name: code_explanation_agent,
    challenge_generator_agent.name: challenge_generator_agent,
    mcq_agent.name: mcq_agent,
}

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

supervisor_routing_tools = [
    route_to_socratic_question,
    route_to_code_analysis,
    route_to_code_explanation,
    route_to_challenge_generator,
    route_to_mcq_generator
]

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
Current sub_topic: {sub_topic}
"""

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", supervisor_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
supervisor_runnable = supervisor_prompt | llm.bind_tools(supervisor_routing_tools)

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
Current sub_topic: {sub_topic}
User struggle count: {user_struggle_count}
MCQ active: {mcq_active}
"""

socratic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", socratic_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
socratic_agent_runnable = socratic_prompt | llm

# --- 3. Define the Graph Nodes ---

def call_supervisor(state: SocraticAgentState):
    messages = state["messages"][-MAX_MESSAGES_IN_CONTEXT:]
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

        if isinstance(tool_call_item, dict):
            tool_name = tool_call_item.get("name", None)
            tool_input = tool_call_item.get("args", {})
        elif hasattr(tool_call_item, "function"):
            tool_name = tool_call_item.function.name
            tool_input = tool_call_item.function.arguments
        else:
            tool_name = None

    if not tool_name:
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
    messages = state["messages"][-MAX_MESSAGES_IN_CONTEXT:]
    response = socratic_agent_runnable.invoke({
        "messages": messages,
        "difficulty_level": state["difficulty_level"],
        "user_struggle_count": state["user_struggle_count"],
        "topic": state["topic"],
        "sub_topic": state["sub_topic"],
        "mcq_active": state["mcq_active"]
    })
    return {"messages": [response], "agent_thought": ""}

def code_analysis_node(state: SocraticAgentState):
    tool_name = "code_analysis_agent"
    tool_args = state["tool_input"]
    tool_function = user_facing_tools_map.get(tool_name)
    tool_output = ""
    if tool_function:
        try:
            tool_output = tool_function.invoke(tool_args)
        except Exception as e:
            tool_output = f"Error executing tool {tool_name}: {e}"
    else:
        tool_output = f"Error: Specialized tool '{tool_name}' not found."
    return {"messages": [ToolMessage(content=tool_output, name=tool_name, tool_call_id=str(uuid.uuid4()))]}

def code_explanation_node(state: SocraticAgentState):
    tool_name = "code_explanation_agent"
    tool_args = state["tool_input"]
    tool_function = user_facing_tools_map.get(tool_name)
    tool_output = ""
    if tool_function:
        try:
            tool_output = tool_function.invoke(tool_args)
        except Exception as e:
            tool_output = f"Error executing tool {tool_name}: {e}"
    else:
        tool_output = f"Error: Specialized tool '{tool_name}' not found."
    return {"messages": [ToolMessage(content=tool_output, name=tool_name, tool_call_id=str(uuid.uuid4()))]}

def challenge_generator_node(state: SocraticAgentState):
    tool_name = "challenge_generator_agent"
    tool_args = state["tool_input"]
    tool_function = user_facing_tools_map.get(tool_name)
    tool_output = ""
    if tool_function:
        try:
            tool_output = tool_function.invoke(tool_args)
        except Exception as e:
            tool_output = f"Error executing tool {tool_name}: {e}"
    else:
        tool_output = f"Error: Specialized tool '{tool_name}' not found."
    return {"messages": [ToolMessage(content=tool_output, name=tool_name, tool_call_id=str(uuid.uuid4()))]}

def generate_mcq_node(state: SocraticAgentState):
    tool_name = "mcq_agent"
    tool_args = state["tool_input"]
    tool_function = user_facing_tools_map.get(tool_name)
    mcq_raw_output = ""

    if tool_function:
        try:
            mcq_raw_output = tool_function.invoke(tool_args)
            mcq_data = json.loads(mcq_raw_output)
            state["mcq_active"] = True
            state["mcq_question"] = mcq_data["question"]
            state["mcq_options"] = mcq_data["options"]
            state["mcq_correct_answer"] = mcq_data["correct_answer"]
        except Exception as e:
            mcq_raw_output = f"Error generating MCQ: {e}"
    
    return {"messages": [ToolMessage(content=mcq_raw_output, name=tool_name, tool_call_id=str(uuid.uuid4()))], **state}

# --- 4. Define the Graph Edges (Conditional Logic) ---

def route_supervisor_output(state: SocraticAgentState):
    if state["next_node"] == "socratic_question":
        return "socratic_question_node"
    elif state["next_node"] == "mcq_generator":
        return "generate_mcq_node"
    elif state["next_node"] == "code_analysis":
        return "code_analysis_node"
    elif state["next_node"] == "code_explanation":
        return "code_explanation_node"
    elif state["next_node"] == "challenge_generator":
        return "challenge_generator_node"
    return "socratic_question_node"

# --- 5. Build the LangGraph ---

workflow = StateGraph(SocraticAgentState)

workflow.add_node("call_supervisor", call_supervisor)
workflow.add_node("socratic_question_node", socratic_question_node)
workflow.add_node("code_analysis_node", code_analysis_node)
workflow.add_node("code_explanation_node", code_explanation_node)
workflow.add_node("challenge_generator_node", challenge_generator_node)
workflow.add_node("generate_mcq_node", generate_mcq_node)

workflow.set_entry_point("call_supervisor")

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

workflow.add_edge("code_analysis_node", "socratic_question_node")
workflow.add_edge("code_explanation_node", "socratic_question_node")
workflow.add_edge("challenge_generator_node", "socratic_question_node")

workflow.add_edge("socratic_question_node", END)
workflow.add_edge("generate_mcq_node", END)

socratic_graph = workflow.compile()

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Socratic Python Tutor")

# Custom CSS for dark theme and styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: #1e1e1e;
        color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #2d2d2d;
        color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        -webkit-transition-duration: 0.4s; /* Safari */
        transition-duration: 0.4s;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        background-color: #3a3a3a;
        color: #f0f2f6;
        border-radius: 8px;
        border: 1px solid #555;
        padding: 10px;
    }
    .stChatMessage {
        background-color: #3a3a3a;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #007bff; /* Blue for user messages */
        color: white;
        align-self: flex-end;
    }
    .stChatMessage.assistant {
        background-color: #3a3a3a; /* Darker grey for assistant messages */
        color: #f0f2f6;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 200px); /* Adjust height as needed */
        overflow-y: auto;
        padding: 10px;
    }
    .st-emotion-cache-1c7y2qn { /* Target Streamlit's main content area */
        padding-top: 2rem;
        padding-right: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
    }
    .st-emotion-cache-1lcbm9l { /* Target Streamlit's sidebar */
        background-color: #2d2d2d;
    }
    .st-emotion-cache-1av5qgq { /* Target Streamlit's main block container */
        gap: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if "socratic_state" not in st.session_state:
    st.session_state.socratic_state = SocraticAgentState(
        messages=[AIMessage(content="Hello! I'm your Socratic Python Tutor. What Python topic would you like to learn or practice today? Or would you like to test your knowledge with a challenge or an MCQ?")],
        difficulty_level="beginner",
        user_struggle_count=0,
        topic="Python Basics",
        sub_topic="Introduction",
        mcq_active=False,
        mcq_question="",
        mcq_options=[],
        mcq_correct_answer="",
        agent_thought="",
        next_node="",
        tool_input={}
    )

def reset_chat():
    st.session_state.socratic_state = SocraticAgentState(
        messages=[AIMessage(content="Hello! I'm your Socratic Python Tutor. What Python topic would you like to learn or practice today? Or would you like to test your knowledge with a challenge or an MCQ?")],
        difficulty_level="beginner",
        user_struggle_count=0,
        topic="Python Basics",
        sub_topic="Introduction",
        mcq_active=False,
        mcq_question="",
        mcq_options=[],
        mcq_correct_answer="",
        agent_thought="",
        next_node="",
        tool_input={}
    )

# Sidebar for Tutor Settings
with st.sidebar:
    st.header("Tutor Settings")
    st.write(f"**Current Difficulty:** {st.session_state.socratic_state['difficulty_level'].capitalize()}")
    st.write(f"**Current Topic:** {st.session_state.socratic_state['topic']}")
    st.write(f"**Current Sub-topic:** {st.session_state.socratic_state['sub_topic']}")
    st.write(f"**Struggle Count:** {st.session_state.socratic_state['user_struggle_count']}")
    if st.button("Clear Chat and Reset"):
        reset_chat()
        st.rerun()

# Main content area
st.write("Streamlit app is running!")
if os.getenv("GOOGLE_API_KEY"):
    st.write("API Key loaded successfully.")
else:
    st.warning("GOOGLE_API_KEY not found. Please set it in your environment variables or .env file.")

st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 50px;">💡</span>
        <h1 style="display: inline-block;">Socratic Python Tutor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Chat history display
chat_container = st.container(height=500, border=True)
with chat_container:
    for message in st.session_state.socratic_state["messages"]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, ToolMessage):
            # Display tool messages for debugging/transparency if needed
            # For a cleaner UI, you might choose not to display raw ToolMessages
            with st.chat_message("assistant"):
                st.markdown(f"*(Tool Output: {message.name})* {message.content}")

# MCQ handling
if st.session_state.socratic_state["mcq_active"]:
    mcq_q = st.session_state.socratic_state["mcq_question"]
    mcq_opts = st.session_state.socratic_state["mcq_options"]
    correct_ans = st.session_state.socratic_state["mcq_correct_answer"]

    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(f"**MCQ:** {mcq_q}")
            selected_option = st.radio("Choose your answer:", mcq_opts, key="mcq_radio")

            if st.button("Submit Answer", key="submit_mcq"):
                if selected_option and mcq_opts.index(selected_option) == (ord(correct_ans.upper()) - ord('A')):
                    st.success("Correct! Well done.")
                    st.session_state.socratic_state["messages"].append(HumanMessage(content=f"My answer for MCQ was: {selected_option}"))
                    st.session_state.socratic_state["messages"].append(AIMessage(content="That's correct! What's next on your learning journey?"))
                    st.session_state.socratic_state["mcq_active"] = False # Deactivate MCQ
                    st.rerun()
                else:
                    st.error(f"Incorrect. The correct answer was {correct_ans}. Let's review the concept. Why do you think '{selected_option}' might be incorrect?")
                    st.session_state.socratic_state["user_struggle_count"] += 1
                    st.session_state.socratic_state["messages"].append(HumanMessage(content=f"My answer for MCQ was: {selected_option}"))
                    st.session_state.socratic_state["messages"].append(AIMessage(content=f"Incorrect. The correct answer was {correct_ans}. Let's review the concept. Why do you think '{selected_option}' might be incorrect?"))
                    st.session_state.socratic_state["mcq_active"] = False # Deactivate MCQ after feedback
                    st.rerun()

# Chat input at the bottom
user_input = st.chat_input("Your message:")

if user_input:
    # Append user message to state
    st.session_state.socratic_state["messages"].append(HumanMessage(content=user_input))
    
    # Run the graph
    try:
        # Create a new state dictionary for the graph execution to avoid modifying
        # st.session_state.socratic_state directly before the graph returns.
        # LangGraph updates the state in place, so we pass a mutable copy.
        current_state_for_graph = st.session_state.socratic_state.copy()
        
        # Ensure the 'messages' list is a mutable list, not an Annotated type
        current_state_for_graph['messages'] = list(current_state_for_graph['messages'])

        for s in socratic_graph.stream(current_state_for_graph):
            for key, value in s.items():
                if key == '__end__':
                    continue
                # Update the session state with the new values from the graph stream
                for k, v in value.items():
                    if k == 'messages':
                        # Append new messages, ensuring they are BaseMessage objects
                        for msg in v:
                            if isinstance(msg, BaseMessage):
                                st.session_state.socratic_state['messages'].append(msg)
                            elif isinstance(msg, dict) and 'content' in msg and 'type' in msg:
                                if msg['type'] == 'human':
                                    st.session_state.socratic_state['messages'].append(HumanMessage(content=msg['content']))
                                elif msg['type'] == 'ai':
                                    st.session_state.socratic_state['messages'].append(AIMessage(content=msg['content']))
                                elif msg['type'] == 'tool':
                                    st.session_state.socratic_state['messages'].append(ToolMessage(content=msg['content'], name=msg.get('name'), tool_call_id=msg.get('tool_call_id', str(uuid.uuid4()))))
                            else:
                                # Fallback for unexpected message types, convert to string
                                st.session_state.socratic_state['messages'].append(AIMessage(content=str(msg)))
                    else:
                        st.session_state.socratic_state[k] = v
        
        # After streaming, ensure the MCQ state is correctly reflected
        # The generate_mcq_node updates mcq_active, mcq_question, etc. in the state.
        # This state is then propagated back to st.session_state.socratic_state.
        # No additional logic needed here unless there are specific post-processing steps.

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.socratic_state["messages"].append(AIMessage(content=f"I encountered an error: {e}. Please try again."))
    
    st.rerun()

