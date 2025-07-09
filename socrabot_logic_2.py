# socratic_bot_logic.py
import os
from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json

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

# --- 2. Initialize the Socratic LLM and Tools ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# New LLM for generating MCQs specifically. Can be the same as socratic_llm or a different one.
# It's good practice to have a separate instance if its system prompt or temperature needs to be different.
mcq_generation_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)

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
    * `code_analysis_agent`: Use this when the user provides code and asks for feedback.
    * `code_explanation_agent`: Use this when the user asks for an explanation.
    * `challenge_generator_agent`: Use this when the user wants a coding challenge.
    * `mcq_agent`: Use this when you want to generate a multiple-choice question for **well-known or predefined topics** (like "variables", "functions", "classes"). This tool has pre-built questions.
    * `llm_mcq_generator`: Use this when the user asks for an MCQ on a topic that is **not explicitly covered by the `mcq_agent`'s predefined list**, or if you believe a more custom or nuanced question is needed based on the current discussion. This tool will ask the LLM to create a new MCQ.
    * `mcq_answer_processor`: Use this tool when the user submits an answer to an active MCQ. Provide the user's answer and the correct answer to this tool. This tool will handle updating the struggle count and resetting the MCQ state.
6.  **Maintain Context:** Keep track of the current topic and sub_topic.
7.  **Be Patient and Encouraging:** Foster a positive learning environment.
8.  **ReAct Architecture:** Before responding or calling a tool, always articulate your thought process. Start your response with "Thought: [Your reasoning here]". Then, proceed with your question or tool call. If you are calling a tool, the tool call should follow your thought. If you are directly asking a question, the question should follow your thought.

Current difficulty level: {difficulty_level}
Current topic: {topic}
Current sub_topic: {sub_topic}
User struggle count: {user_struggle_count}
MCQ active: {mcq_active}
MCQ Question (internal): {mcq_question} # Note: This is now the formatted string
MCQ Options (internal): {mcq_options}
MCQ Correct Answer (internal): {mcq_correct_answer}

Begin the conversation by asking the user what Python topic they'd like to learn or practice, or if they'd like to test their knowledge.
"""

socratic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", socratic_system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

@tool
def code_analysis_agent(code: str) -> str:
    """Analyzes the provided Python code..."""
    return f"Simulated Code Analysis: Your code snippet '{code}' looks interesting. What were you trying to achieve with this code?"

@tool
def code_explanation_agent(concept: str) -> str:
    """Explains a given Python concept..."""
    return f"Simulated Code Explanation: Ah, you're curious about '{concept}'. Can you tell me what you already know or suspect about it?"

@tool
def challenge_generator_agent(topic: str, difficulty: str) -> str:
    """Generates a Python coding challenge..."""
    return f"Simulated Challenge for '{topic}': 'Write a function that sums even numbers in a list.' How would you start?"

@tool
def mcq_agent(topic: str, difficulty: str) -> str:
    """
    Generates a multiple-choice question (MCQ) on a given Python topic and difficulty level.
    The output will be a JSON string containing the question, options, and correct answer.
    The 'question' field will be pre-formatted to include options for direct display.
    This tool is called when the Socratic agent decides to test understanding via MCQ.
    """
    mcqs_raw = {
        "variables": {
            "question": "Which of the following data types is mutable in Python?",
            "options": ["A) Tuple", "B) String", "C) List", "D) Integer"],
            "correct_answer": "C"
        },
        "class": {
            "question": "In Python, what is the primary purpose of the `__init__` method in a class?",
            "options": [
                "A) To destroy an object when it's no longer needed.",
                "B) To define static methods.",
                "C) To initialize the attributes of an object when it's created.",
                "D) To define the string representation of an object."
            ],
            "correct_answer": "C"
        },
        "functions": {
            "question": "Which keyword is used to define a function in Python?",
            "options": ["A) func", "B) define", "C) def", "D) function"],
            "correct_answer": "C"
        },
        "conditional statements": {
            "question": "Which Python keyword is used to start an 'if' statement?",
            "options": ["A) then", "B) if", "C) when", "D) check"],
            "correct_answer": "B"
        },
        "comparisons": {
            "question": "What is the correct operator for 'not equal to' in Python?",
            "options": ["A) ==", "B) !=", "C) <>", "D) =!"],
            "correct_answer": "B"
        },
        "maximum of three numbers": {
            "question": "Consider finding the maximum of three numbers (a, b, c). Which of these logical structures is typically used?",
            "options": [
                "A) A single 'for' loop",
                "B) Nested 'if-else' statements or multiple 'if' statements with logical 'and'/'or'",
                "C) A 'while' loop",
                "D) A 'try-except' block"
            ],
            "correct_answer": "B"
        }
    }
    
    # Check if the exact topic exists in our predefined list
    selected_mcq_raw = mcqs_raw.get(topic.lower())

    if selected_mcq_raw:
        # Format the question to include options for direct display in chat
        formatted_question = f"**{selected_mcq_raw['question']}**\n\n" + \
                             "\n".join(selected_mcq_raw['options'])

        mcq_data = {
            "question": formatted_question,
            "options": selected_mcq_raw['options'],
            "correct_answer": selected_mcq_raw['correct_answer']
        }
        return json.dumps(mcq_data)
    else:
        # If topic not found, return an indicator so the LLM knows to use llm_mcq_generator
        # This is a bit of a hack but tells the LLM "I don't have this"
        return "NO_PREDEFINED_MCQ_FOUND"


@tool
def llm_mcq_generator(topic: str, difficulty: str) -> str:
    """
    Generates a new multiple-choice question (MCQ) on a given Python topic and difficulty level
    by directly invoking an LLM.
    The output will be a JSON string containing the question, options (A, B, C, D), and correct answer (A, B, C, or D).
    Example JSON format:
    {
        "question": "What is the purpose of a 'for' loop in Python?",
        "options": ["A) To define a function.", "B) To iterate over a sequence.", "C) To handle exceptions.", "D) To create a new class."],
        "correct_answer": "B"
    }
    """
    mcq_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert Python programming instructor. Your task is to generate a challenging
        but fair multiple-choice question (MCQ) on the given Python topic and difficulty level.
        The MCQ should have exactly four options (A, B, C, D).
        Provide the question, the four options, and the single correct answer in a JSON format.
        Ensure the question and options are clear and unambiguous.
        The correct answer must be one of A, B, C, or D.

        Example Output:
        {
            "question": "What is the purpose of a 'for' loop in Python?",
            "options": ["A) To define a function.", "B) To iterate over a sequence.", "C) To handle exceptions.", "D) To create a new class."],
            "correct_answer": "B"
        }
        """),
        ("user", f"Generate a Python MCQ about: {topic}. Difficulty: {difficulty}.")
    ])

    try:
        response = mcq_generation_llm.invoke(mcq_prompt_template.format(topic=topic, difficulty=difficulty))
        mcq_data = json.loads(response.content)

        # Basic validation of the LLM's output
        if not all(k in mcq_data for k in ["question", "options", "correct_answer"]):
            raise ValueError("MCQ JSON missing required keys.")
        if not isinstance(mcq_data["options"], list) or len(mcq_data["options"]) != 4:
            raise ValueError("MCQ options must be a list of 4 items.")
        if mcq_data["correct_answer"] not in ["A", "B", "C", "D"]:
            raise ValueError("Correct answer must be A, B, C, or D.")

        # Format the question to include options for direct display
        formatted_question = f"**{mcq_data['question']}**\n\n" + \
                             "\n".join(mcq_data['options'])

        # Return the formatted data including question (for display) and raw options/answer (for logic)
        return json.dumps({
            "question": formatted_question,
            "options": mcq_data['options'],
            "correct_answer": mcq_data['correct_answer']
        })

    except json.JSONDecodeError:
        return json.dumps({"error": "Failed to decode JSON from LLM for MCQ generation."})
    except ValueError as e:
        return json.dumps({"error": f"Invalid MCQ format generated by LLM: {e}"})
    except Exception as e:
        return json.dumps({"error": f"An unexpected error occurred during MCQ generation: {e}"})


@tool
def mcq_answer_processor(user_answer: str, correct_answer: str) -> str:
    """
    Processes the user's answer to an MCQ. Updates struggle count and resets MCQ state.
    Returns feedback message (e.g., "Correct!" or "Incorrect. Let's review.")
    """
    is_correct = user_answer.strip().upper() == correct_answer.strip().upper()
    if is_correct:
        return "Correct!"
    else:
        return "Incorrect."


tools = [code_analysis_agent, code_explanation_agent, challenge_generator_agent, mcq_agent, llm_mcq_generator, mcq_answer_processor]
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")
socratic_agent_runnable = socratic_prompt | llm_with_tools

# --- 3. Define the Graph Nodes ---

def call_llm(state: SocraticAgentState):
    """Invokes the LLM with the current conversation history."""
    print("[DEBUG] Messages sent to LLM:", state["messages"])
    response = socratic_agent_runnable.invoke({
        "messages": state["messages"],
        **{k: v for k, v in state.items() if k != 'messages'} # Pass all other state variables to the prompt
    })
    print("[DEBUG] LLM Response:", response)
    
    thought = ""
    display_content = response.content # Start with the full content

    if response.content and response.content.startswith("Thought:"):
        # Extract the thought for internal use
        parts = response.content.split("Thought:", 1)
        thought = parts[1].strip().split('\n', 1)[0] # Get the first line of the thought

        # Extract the content to be displayed to the user
        if '\n' in parts[1]:
            display_content = parts[1].split('\n', 1)[1].strip()
        else:
            display_content = "" # If thought is the only content, hide it

    new_ai_message = AIMessage(
        content=display_content,
        tool_calls=response.tool_calls # Preserve tool calls if any
    )

    return {"messages": [new_ai_message], "agent_thought": thought}

TOOLS_USED = {
    "code_analysis_agent": code_analysis_agent,
    "code_explanation_agent": code_explanation_agent,
    "challenge_generator_agent": challenge_generator_agent,
    "mcq_agent": mcq_agent,
    "llm_mcq_generator": llm_mcq_generator, # Add the new tool here
    "mcq_answer_processor": mcq_answer_processor,
}

def call_tool(state: SocraticAgentState):
    """Executes a tool call and returns only the updated state fields."""
    last_message = state["messages"][-1]
    
    messages_to_add = []
    state_updates = {}

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_function = TOOLS_USED.get(tool_name)
            
            tool_output_content = ""
            if tool_function:
                response = tool_function.invoke(tool_args)
                tool_output_content = str(response)

                # Handle output from both MCQ generation tools
                if tool_name in ["mcq_agent", "llm_mcq_generator"]:
                    if tool_output_content == "NO_PREDEFINED_MCQ_FOUND":
                        # This specific response from mcq_agent tells us to let the LLM decide
                        # to call llm_mcq_generator in the next step.
                        # Do not update state_updates here for MCQ, let the LLM re-plan.
                        messages_to_add.append(
                            ToolMessage(content="No predefined MCQ found for this topic.", tool_call_id=tool_call["id"])
                        )
                        # We don't set mcq_active etc. here, because the LLM needs to make another decision.
                        continue # Skip to next tool call or finish this loop
                    
                    try:
                        mcq_data = json.loads(tool_output_content)
                        # Check for error from llm_mcq_generator
                        if "error" in mcq_data:
                             messages_to_add.append(
                                ToolMessage(content=f"Error generating MCQ: {mcq_data['error']}", tool_call_id=tool_call["id"])
                             )
                             continue # Skip to next tool call or finish this loop

                        state_updates["mcq_active"] = True
                        state_updates["mcq_question"] = mcq_data.get("question", "")
                        state_updates["mcq_options"] = mcq_data.get("options", [])
                        state_updates["mcq_correct_answer"] = mcq_data.get("correct_answer", "")
                        if not state.get("topic") and tool_args.get("topic"):
                            state_updates["topic"] = tool_args["topic"]
                    except json.JSONDecodeError:
                        messages_to_add.append(
                            ToolMessage(content="Error: MCQ agent returned invalid JSON.", tool_call_id=tool_call["id"])
                        )
                        continue # Skip to next tool call or finish this loop

                elif tool_name == "mcq_answer_processor":
                    if tool_output_content == "Correct!":
                        state_updates["user_struggle_count"] = 0
                    else:
                        state_updates["user_struggle_count"] = state.get("user_struggle_count", 0) + 1
                    
                    state_updates["mcq_active"] = False
                    state_updates["mcq_question"] = ""
                    state_updates["mcq_options"] = []
                    state_updates["mcq_correct_answer"] = ""
                
                # For other tools, just add their output as a ToolMessage
                messages_to_add.append(
                    ToolMessage(content=tool_output_content, tool_call_id=tool_call["id"])
                )

            else:
                messages_to_add.append(
                    ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call["id"])
                )

    return {"messages": messages_to_add, **state_updates}


# --- 4. Define the Graph Edges ---

def should_continue(state: SocraticAgentState):
    """Determines the next step in the graph."""
    if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls:
        return "call_tool"
    return "end"

# --- 5. Build the LangGraph ---

workflow = StateGraph(SocraticAgentState)

workflow.add_node("call_llm", call_llm)
workflow.add_node("call_tool", call_tool)

workflow.set_entry_point("call_llm")

workflow.add_conditional_edges(
    "call_llm",
    should_continue,
    {"call_tool": "call_tool", "end": END}
)
workflow.add_edge("call_tool", "call_llm")

socratic_graph = workflow.compile()

# --- Test ---
try:
    from IPython.display import Image, display
    display(Image(socratic_graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# Example test for the new LLM-generated MCQ
print("\n--- Testing LLM-generated MCQ for an unknown topic ---")
llm_mcq_state = {
    "messages": [HumanMessage(content="Can you give me an MCQ on Python decorators?")],
    "difficulty_level": "intermediate",
    "user_struggle_count": 0,
    "topic": "Python decorators",
    "sub_topic": "",
    "mcq_active": False,
    "mcq_question": "",
    "mcq_options": [],
    "mcq_correct_answer": "",
    "agent_thought": ""
}
result_llm_mcq = socratic_graph.invoke(llm_mcq_state)
import pprint
pprint.pprint(result_llm_mcq)

# Example test for a known MCQ topic
# print("\n--- Testing Predefined MCQ for 'variables' ---")
# known_mcq_state = {
#     "messages": [HumanMessage(content="Give me an MCQ on graphs.")],
#     "difficulty_level": "beginner",
#     "user_struggle_count": 0,
#     "topic": "variables",
#     "sub_topic": "",
#     "mcq_active": False,
#     "mcq_question": "",
#     "mcq_options": [],
#     "mcq_correct_answer": "",
#     "agent_thought": ""
# }
# result_known_mcq = socratic_graph.invoke(known_mcq_state)
# pprint.pprint(result_known_mcq)