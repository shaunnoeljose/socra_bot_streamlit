# terminal_test.py

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, ToolCall # Import ToolCall
from socratic_bot_logic import socratic_graph, SocraticAgentState # Import socratic_graph (now compiled with checkpointer)
import uuid # To generate unique thread IDs

# Load environment variables from .env file
load_dotenv()

# Ensure GOOGLE_API_KEY is set directly
# For testing, you must ensure GOOGLE_API_KEY is set in your system environment or .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set. Please set it in a .env file or your system environment.")
    exit()

print("Socratic Python Tutor (Terminal Mode)")
print("Type 'exit' to end the conversation.")
print("Type 'new' to start a new conversation thread.")
print("---")

# --- MemorySaver Integration ---
thread_id = "my_socratic_test_session" # Fixed ID for persistent testing

# Initial state for the graph.
initial_graph_state = SocraticAgentState(
    messages=[],
    difficulty_level="beginner",
    user_struggle_count=0,
    topic="Python Basics",
    sub_topic="Introduction",
    mcq_active=False,
    mcq_question="",
    mcq_options=[],
    mcq_correct_answer="",
    agent_thought=""
)

# Function to convert a message dictionary back to a BaseMessage object
# This is crucial for consistent attribute access, including nested ToolCalls
def _convert_to_message_object(msg_dict: dict) -> BaseMessage:
    msg_type = msg_dict.get('type')
    if msg_type == 'human':
        return HumanMessage(**msg_dict)
    elif msg_type == 'ai':
        # Special handling for AIMessage to convert nested tool_calls
        tool_calls_raw = msg_dict.get('tool_calls')
        if tool_calls_raw:
            converted_tool_calls = []
            for tc_raw in tool_calls_raw:
                # Ensure tc_raw is a dict, then access its components
                if isinstance(tc_raw, dict):
                    function_dict = tc_raw.get('function', {})
                    converted_tool_calls.append(
                        ToolCall(
                            id=tc_raw.get('id'), # Preserve ID if present
                            function={"name": function_dict.get('name'), "arguments": function_dict.get('arguments')}
                        )
                    )
                else: # If it's already a ToolCall object (e.g., newly created), just append it
                    converted_tool_calls.append(tc_raw)

            # Create a new dictionary for AIMessage constructor, ensuring tool_calls is the converted list
            # Remove original 'tool_calls' from msg_dict if present, then add the converted list
            temp_msg_dict = {k: v for k, v in msg_dict.items() if k != 'tool_calls'}
            return AIMessage(tool_calls=converted_tool_calls, **temp_msg_dict)
        return AIMessage(**msg_dict)
    elif msg_type == 'tool':
        return ToolMessage(**msg_dict)
    else:
        # Fallback for unexpected types, or if 'type' key is missing
        return BaseMessage(content=str(msg_dict.get('content', ''))) # Safely get content

# Load the existing state for the thread_id, or initialize if it's new.
try:
    print(f"Attempting to load conversation for thread ID: {thread_id}")
    loaded_state_checkpoint = socratic_graph.get_state(config={"configurable": {"thread_id": thread_id}})
    
    if loaded_state_checkpoint and loaded_state_checkpoint.values:
        current_state = loaded_state_checkpoint.values # Get the actual state dictionary
        print(f"\n--- DEBUG: State after initial load (raw dicts): {current_state['messages']} ---\n")
        # Convert all messages in the loaded state back to BaseMessage objects
        current_state["messages"] = [_convert_to_message_object(msg) if isinstance(msg, dict) else msg for msg in current_state["messages"]]
        print(f"\n--- DEBUG: State after initial load (converted objects): {current_state['messages']} ---\n")
        print("Loaded previous conversation state.")
    else:
        current_state = initial_graph_state
        print("No previous conversation found. Starting fresh.")

except Exception as e:
    print(f"Error loading previous state for thread ID {thread_id}: {e}")
    print("Starting a new conversation.")
    current_state = initial_graph_state

# Initial greeting
if not current_state["messages"]:
    initial_message = "Hello! I'm your Socratic Python Tutor. What Python topic would you like to learn or practice today? Or would you like to test your knowledge with a challenge or an MCQ?"
    print(f"Assistant: {initial_message}")
    current_state["messages"].append(AIMessage(content=initial_message))
else:
    print("Resuming previous conversation:")
    # Iterate through messages, now guaranteed to be BaseMessage objects
    for msg in current_state["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"You: {msg.content}")
        elif isinstance(msg, AIMessage):
            display_content = msg.content
            if display_content and display_content.startswith("Thought:"):
                thought_end_index = display_content.find('\n')
                if thought_end_index != -1:
                    display_content = display_content[thought_end_index:].strip()
                else:
                    display_content = "" # If only thought, display nothing
            if display_content:
                print(f"Assistant: {display_content}")
            
            if msg.tool_calls:
                # Access tool name directly from ToolCall object
                # This is the line that was causing the error - now msg.tool_calls[0] should be a ToolCall object
                tool_name_to_display = msg.tool_calls[0].function.name
                print(f"Assistant: (Tool Call: {tool_name_to_display})")
        elif isinstance(msg, ToolMessage):
            print(f"Assistant (Tool Output from {msg.name}): {msg.content}")
        else:
            print(f"Unknown message type in history: {msg}")


while True:
    try:
        if current_state["mcq_active"]:
            print("\n--- Multiple Choice Question ---")
            print(current_state["mcq_question"])
            for i, option in enumerate(current_state["mcq_options"]):
                print(f"{chr(65 + i)}) {option}")
            user_answer = input("Your answer (e.g., A, B, C): ").strip().upper()

            if user_answer == 'EXIT':
                print("Exiting Socratic Tutor. Goodbye!")
                break
            if user_answer.lower() == 'new':
                print("Starting a new conversation thread.")
                thread_id = str(uuid.uuid4()) # Generate a new UUID for a new thread
                current_state = initial_graph_state # Reset state for new thread
                print(f"New conversation thread ID: {thread_id}")
                # Re-display initial greeting for new thread
                initial_message = "Hello! I'm your Socratic Python Tutor. What Python topic would you like to learn or practice today? Or would you like to test your knowledge with a challenge or an MCQ?"
                print(f"Assistant: {initial_message}")
                current_state["messages"].append(AIMessage(content=initial_message))
                continue


            correct_answer_key = current_state["mcq_correct_answer"]
            is_correct = (user_answer == correct_answer_key)

            feedback_message = ""
            if is_correct:
                feedback_message = f"That's correct! The answer is {correct_answer_key}) {current_state['mcq_options'][ord(correct_answer_key) - ord('A')]}. Excellent!"
                current_state["user_struggle_count"] = 0
            else:
                feedback_message = f"That's not quite right. The correct answer was {correct_answer_key}) {current_state['mcq_options'][ord(correct_answer_key) - ord('A')]}. Let's review this concept."
                current_state["user_struggle_count"] += 1

            print(f"Assistant: {feedback_message}")
            current_state["messages"].append(AIMessage(content=feedback_message))

            # Reset MCQ state
            current_state["mcq_active"] = False
            current_state["mcq_question"] = ""
            current_state["mcq_options"] = []
            current_state["mcq_correct_answer"] = ""
            # After MCQ, we want the Socratic agent to continue the conversation
            # We'll feed the feedback back into the graph in the next invocation.
            # No `continue` here, as we want to invoke the graph with the feedback.

        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            print("Exiting Socratic Tutor. Goodbye!")
            break
        if user_input.lower() == 'new':
            print("Starting a new conversation thread.")
            thread_id = str(uuid.uuid4()) # Generate a new UUID for a new thread
            current_state = initial_graph_state # Reset state for new thread
            print(f"New conversation thread ID: {thread_id}")
            # Re-display initial greeting for new thread
            initial_message = "Hello! I'm your Socratic Python Tutor. What Python topic would you like to learn or practice today? Or would you like to test your knowledge with a challenge or an MCQ?"
            print(f"Assistant: {initial_message}")
            current_state["messages"].append(AIMessage(content=initial_message))
            continue # Skip graph invocation for this turn, go to next loop iteration

        current_state["messages"].append(HumanMessage(content=user_input))

        # Invoke the graph with the thread_id in the config
        final_state = socratic_graph.invoke(
            current_state,
            config={"configurable": {"thread_id": thread_id}}
        )

        # Update the current state with the final state from the graph
        current_state = final_state
        print(f"\n--- DEBUG: State after invoke (raw dicts): {current_state['messages']} ---\n")
        # Crucial: Convert all messages in the state to BaseMessage objects after invoke
        current_state["messages"] = [_convert_to_message_object(msg) if isinstance(msg, dict) else msg for msg in current_state["messages"]]
        print(f"\n--- DEBUG: State after invoke (converted objects): {current_state['messages']} ---\n")


        # Removed logging of agent_thought as per request
        # if current_state["agent_thought"]:
        #     logger.info(f"Agent Thought: {current_state['agent_thought']}")

        # Extract the last AI message for display
        last_ai_message = None
        for msg in reversed(current_state["messages"]):
            if isinstance(msg, AIMessage): # Now we expect it to be AIMessage object
                last_ai_message = msg
                break

        if last_ai_message:
            display_content = last_ai_message.content
            # Strip "Thought:" prefix for display if it exists
            if display_content and display_content.startswith("Thought:"):
                thought_end_index = display_content.find('\n')
                if thought_end_index != -1:
                    display_content = display_content[thought_end_index:].strip()
                else:
                    display_content = ""

            if display_content:
                print(f"Assistant: {display_content}")
            
            # This block is for displaying tool call initiation and tool output in terminal
            # It will only print if a tool was called and its output was captured as ToolMessage
            if last_ai_message.tool_calls:
                # Access tool name directly from ToolCall object
                tool_name_to_display = last_ai_message.tool_calls[0].function.name
                print(f"Assistant: (Tool Call Initiated: {tool_name_to_display})")
                
                # Find the corresponding ToolMessage object
                tool_output_msg = None
                for m in reversed(current_state["messages"]):
                    if isinstance(m, ToolMessage) and m.name == tool_name_to_display:
                        tool_output_msg = m
                        break

                if tool_output_msg:
                    print(f"Assistant (Tool Output): {tool_output_msg.content}")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Removed logging of errors to logger as per request
        # logger.error(f"Error during interaction: {e}", exc_info=True)
        break
