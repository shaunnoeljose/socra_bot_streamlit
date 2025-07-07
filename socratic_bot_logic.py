# socratic_bot_logic.py

import os
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
# Removed ToolExecutor and ToolInvocation imports
# from langgraph.prebuilt import ToolExecutor, ToolInvocation

# Import the logging utility (assuming logger.py will be created later)
# from logger import setup_logger
# logger = setup_logger()

# --- 1. Define the Agent State ---
# This TypedDict defines the structure of our graph's state.
# It will hold all the information needed to manage the conversation flow.
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
    agent_thought: str # Added for ReAct architecture

# --- 2. Initialize the Socratic LLM and Tools ---

# Initialize the Gemini LLM for the Socratic Agent
# Ensure GOOGLE_API_KEY is set in your environment variables or .env file
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Define the system prompt for the Socratic Agent.
# This prompt guides the LLM's behavior, making it act as a Socratic tutor,
# detecting struggle, adapting difficulty, and using tools.
socratic_system_prompt = """
You are a Socratic Python programming tutor. Your goal is to guide the user to discover answers
and understand concepts through thoughtful questions, rather than directly providing solutions.

Here are your core principles:
1.  **Ask Questions:** Always respond with a question, unless explicitly providing feedback on code or an MCQ answer.
2.  **Socratic Method:** Break down complex problems into smaller, manageable questions.
3.  **Encourage Exploration:** Prompt the user to experiment, research, or think critically.
4.  **Adapt to User Understanding:**
    * **Struggle Detection:** If the user seems confused, provides incorrect answers, or asks for direct solutions, simplify your questions, rephrase, or offer a hint. You can also suggest taking a multiple-choice question (MCQ) to assess their understanding differently.
    * **Progression:** If the user demonstrates understanding, subtly move to a slightly more advanced sub-concept or a related new topic. Avoid repetitive questioning on the same point.
5.  **Tool Usage:** You have access to several specialized tools. Use them judiciously based on the user's query:
    * `code_analysis_agent`: Use this when the user provides Python code and asks for feedback, debugging, or analysis.
    * `code_explanation_agent`: Use this when the user asks for an explanation of a Python concept, function, keyword, or error message.
    * `challenge_generator_agent`: Use this when the user wants a coding challenge or a fill-in-the-blanks exercise.
    * `mcq_agent`: Use this when you want to generate a multiple-choice question to test the user's understanding, especially if they are struggling or you want to quickly assess a concept.
6.  **Maintain Context:** Keep track of the current topic and sub_topic.
7.  **Be Patient and Encouraging:** Foster a positive learning environment.
8.  **ReAct Architecture:** Before responding or calling a tool, always articulate your thought process. Start your response with "Thought: [Your reasoning here]". Then, proceed with your question or tool call. If you are calling a tool, the tool call should follow your thought. If you are directly asking a question, the question should follow your thought.

Current difficulty level: {difficulty_level}
Current topic: {topic}
Current sub_topic: {sub_topic}
User struggle count: {user_struggle_count}
MCQ active: {mcq_active}

Begin the conversation by asking the user what Python topic they'd like to learn or practice, or if they'd like to test their knowledge.
"""

# Create the prompt template for the Socratic LLM, including tool calling capabilities.
socratic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", socratic_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# --- Define Simulated Agent Tools (will be replaced by actual agents later) ---
# These are placeholder tools for now. In a full multi-agent system,
# these would likely be separate LangChain agents or services.

@tool
def code_analysis_agent(code: str) -> str:
    """
    Analyzes the provided Python code, identifies potential issues, suggests improvements,
    and provides feedback. Use this when the user provides code and asks for review or debugging.
    """
    # In a real scenario, this would call another LLM or a static analysis tool.
    return f"Simulated Code Analysis: Your code snippet '{code}' looks interesting. Let's analyze it together. What were you trying to achieve with this code?"

@tool
def code_explanation_agent(concept: str) -> str:
    """
    Explains a given Python concept, function, keyword, or error message in detail.
    Use this when the user asks for an explanation of something.
    """
    # In a real scenario, this would call another LLM specialized in explanations.
    return f"Simulated Code Explanation: Ah, you're curious about '{concept}'. Instead of me explaining it directly, can you tell me what you already know or suspect about '{concept}'?"

@tool
def challenge_generator_agent(topic: str, difficulty: str) -> str:
    """
    Generates a Python coding challenge or a fill-in-the-blanks exercise based on the specified topic and difficulty.
    Use this when the user requests a challenge.
    """
    # In a real scenario, this would call another LLM or a challenge generation service.
    return f"Simulated Challenge: For '{topic}' at '{difficulty}' difficulty, here's a challenge: 'Write a Python function that takes a list of numbers and returns the sum of all even numbers.' How would you approach this?"

@tool
def mcq_agent(topic: str, difficulty: str) -> str:
    """
    Generates a multiple-choice question (MCQ) on a given Python topic and difficulty level.
    The output will be a JSON string containing the question, options, and correct answer.
    This tool is called when the Socratic agent decides to test understanding via MCQ.
    """
    # In a real scenario, this would call another LLM specifically for MCQ generation.
    # For now, we return a simulated JSON string.
    # The main application logic will parse this and present the MCQ.
    mcq_data = {
        "question": f"Which of the following data types is mutable in Python?",
        "options": ["A) Tuple", "B) String", "C) List", "D) Integer"],
        "correct_answer": "C"
    }
    import json
    return json.dumps(mcq_data)

# List of all tools available to the Socratic agent
tools = [code_analysis_agent, code_explanation_agent, challenge_generator_agent, mcq_agent]

# Bind the tools to the LLM first
llm_with_tools = llm.bind_tools(tools)

# Then, create the runnable agent by piping the prompt and the LLM with tools
socratic_agent_runnable = socratic_prompt | llm_with_tools

# --- 3. Define the Graph Nodes ---

def call_llm(state: SocraticAgentState):
    """
    Node to call the Socratic LLM.
    The LLM processes the messages and decides whether to respond directly
    or call a tool. It also articulates its thought process.
    """
    messages = state["messages"]
    response = socratic_agent_runnable.invoke({
        "messages": messages,
        "difficulty_level": state["difficulty_level"],
        "user_struggle_count": state["user_struggle_count"],
        "topic": state["topic"],
        "sub_topic": state["sub_topic"],
        "mcq_active": state["mcq_active"]
    })

    # Extract the thought from the response content if present
    thought = ""
    if response.content and response.content.startswith("Thought:"):
        # Find the end of the thought and the beginning of the actual message/tool call
        parts = response.content.split("Thought:", 1)
        if len(parts) > 1:
            thought_and_rest = parts[1].strip()
            # If there's a subsequent "Action:" or direct question, try to split it
            # For now, we'll just take everything after "Thought:" as the thought.
            # The actual message content (question) will be handled by the main loop.
            thought = thought_and_rest.split('\n', 1)[0] # Take the first line as thought
            # The rest of the content will be part of the AIMessage itself
            # We don't need to modify response.content here as the prompt guides the LLM
            # to put the thought *before* any other content or tool calls.

    return {"messages": [response], "agent_thought": thought}


def call_tool(state: SocraticAgentState):
    """
    Node to execute a tool call requested by the LLM.
    It parses the tool invocation and runs the tool.
    """
    last_message = state["messages"][-1]
    tool_output_messages = []

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            # logger.info(f"Calling tool: {tool_call.name} with args: {tool_call.args}")
            # Instead of ToolInvocation and ToolExecutor, directly call the tool function
            # This assumes the tool functions are directly callable within this scope
            tool_function = globals().get(tool_call.name)
            if tool_function:
                response = tool_function(**tool_call.args)
                tool_output_messages.append(AIMessage(content=str(response), name=tool_call.name))
                # logger.info(f"Tool '{tool_call.name}' output: {response}")

                # Special handling for MCQ agent output
                if tool_call.name == "mcq_agent":
                    import json
                    mcq_data = json.loads(response)
                    state["mcq_active"] = True
                    state["mcq_question"] = mcq_data["question"]
                    state["mcq_options"] = mcq_data["options"]
                    state["mcq_correct_answer"] = mcq_data["correct_answer"]
                    # The actual MCQ presentation and evaluation will happen in main.py
                    # We just update the state here to indicate an MCQ is active.
            else:
                tool_output_messages.append(AIMessage(content=f"Error: Tool '{tool_call.name}' not found.", name=tool_call.name))


    return {"messages": tool_output_messages, **state} # Return updated state including MCQ details

# --- 4. Define the Graph Edges (Conditional Logic) ---

def should_continue(state: SocraticAgentState):
    """
    Determines the next step in the graph based on the last message.
    If the LLM requested a tool, it goes to 'call_tool'. Otherwise, it ends (for now).
    Later, we'll add more complex logic here for MCQ handling, etc.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # If the MCQ agent was called, we need a specific flow for it
        if any(tc.name == "mcq_agent" for tc in last_message.tool_calls):
            return "generate_mcq" # Go to a specific node for MCQ handling
        return "call_tool"
    return "end" # Direct response from LLM, conversation loop will continue in main.py

def generate_mcq_node(state: SocraticAgentState):
    """
    A specific node to handle the output of the MCQ agent.
    This node primarily updates the state to reflect the active MCQ.
    The actual presentation to the user and evaluation will happen outside the graph in main.py.
    """
    # The `call_tool` node already parsed the MCQ output and updated the state.
    # This node primarily serves as a distinct point in the graph for MCQ flow.
    # We can add more specific MCQ-related logic here if needed, but for now,
    # it just confirms the MCQ state is set.
    # logger.info("MCQ generation node activated. MCQ details updated in state.")
    return state # Return the state as is, after call_tool has updated it.

# --- 5. Build the LangGraph ---

# Create a StateGraph instance with our defined state.
workflow = StateGraph(SocraticAgentState)

# Add nodes to the workflow.
workflow.add_node("call_llm", call_llm)
workflow.add_node("call_tool", call_tool)
workflow.add_node("generate_mcq", generate_mcq_node) # Add the MCQ specific node

# Set the entry point for the graph.
workflow.set_entry_point("call_llm")

# Define the edges.
# From 'call_llm', decide if we need to call a tool or end the current graph run.
workflow.add_conditional_edges(
    "call_llm",
    should_continue,
    {
        "call_tool": "call_tool",
        "generate_mcq": "generate_mcq", # If MCQ tool was called, go to generate_mcq node
        "end": END # If no tool call, the LLM's direct response ends this graph run.
    }
)

# After calling a tool or generating an MCQ, we typically want to loop back to the LLM
# to process the tool's output or continue the Socratic questioning.
workflow.add_edge("call_tool", "call_llm")
workflow.add_edge("generate_mcq", END) # After generating MCQ, the graph run ends.
                                     # The main loop will handle user input for MCQ.

# Compile the graph into a runnable agent.
socratic_graph = workflow.compile()

# Print the graph (useful for debugging)
# from IPython.display import Image, display
# try:
#     display(Image(socratic_graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This is a common error if graphviz is not installed.
#     # We can just ignore it for now.
#     pass

# logger.info("Socratic LangGraph workflow compiled.")