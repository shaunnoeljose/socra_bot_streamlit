# main.py

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from socratic_bot_logic import socratic_graph, SocraticAgentState
from logger import logger # Import the configured logger

# Load environment variables from .env file
load_dotenv()

# --- Debugging: Check if Streamlit is rendering at all ---
st.write("Streamlit app is running!")

# Ensure GOOGLE_API_KEY is set
google_api_key_set = bool(os.getenv("GOOGLE_API_KEY"))
if not google_api_key_set:
    st.error("GOOGLE_API_KEY environment variable not set. Please set it in a .env file.")
    st.stop()
else:
    st.write("API Key loaded successfully.") # Debugging line to confirm API key presence

st.set_page_config(page_title="Socratic Python Tutor", page_icon="🐍")
st.title("🐍 Socratic Python Tutor")
st.markdown("---")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "socratic_agent_state" not in st.session_state:
    # Initialize the LangGraph state
    st.session_state.socratic_agent_state = SocraticAgentState(
        messages=[],
        difficulty_level="beginner",
        user_struggle_count=0,
        topic="Python Basics", # Default topic
        sub_topic="Introduction", # Default sub-topic
        mcq_active=False,
        mcq_question="",
        mcq_options=[],
        mcq_correct_answer="",
        agent_thought=""
    )
if "initial_greeting_done" not in st.session_state:
    st.session_state.initial_greeting_done = False
if "mcq_displayed" not in st.session_state: # Ensure this is initialized
    st.session_state.mcq_displayed = False

# --- Display Chat History ---
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
            if message.name: # Display tool outputs
                st.info(f"Tool Output ({message.name}): {message.content}")

# --- Initial Greeting and Topic Selection ---
if not st.session_state.initial_greeting_done:
    initial_message = "Hello! I'm your Socratic Python Tutor. What Python topic would you like to learn or practice today? Or would you like to test your knowledge with a challenge or an MCQ?"
    with st.chat_message("assistant"):
        st.markdown(initial_message)
    st.session_state.chat_history.append(AIMessage(content=initial_message))
    st.session_state.initial_greeting_done = True

# --- Main Chat Input and Processing ---
user_input = st.chat_input("Your message:")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Update the LangGraph state with the new user message
    current_state = st.session_state.socratic_agent_state
    current_state["messages"].append(HumanMessage(content=user_input))

    # Run the LangGraph
    try:
        # The graph will return the updated state after one or more steps
        # We need to pass the current state to the graph
        # The graph's output is the final state after its execution path
        final_state = socratic_graph.invoke(current_state)

        # Update the session state with the final state from the graph
        st.session_state.socratic_agent_state = final_state
        logger.info(f"Updated Socratic Agent State: {st.session_state.socratic_agent_state}")
        # Log the agent's thought to the logger file (this goes to console if LOG_TO_CONSOLE is True, and to file)
        if final_state["agent_thought"]:
            logger.info(f"Agent Thought: {final_state['agent_thought']}")

        # Extract the last message from the updated state for display
        last_ai_message = None
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break

        if last_ai_message:
            with st.chat_message("assistant"):
                # Extract and display only the content after "Thought:" if it exists
                display_content = last_ai_message.content
                if display_content and display_content.startswith("Thought:"):
                    # Find the first newline or the end of the string after "Thought:"
                    # This assumes the thought is on the first line, followed by the actual question.
                    thought_end_index = display_content.find('\n')
                    if thought_end_index != -1:
                        display_content = display_content[thought_end_index:].strip()
                    else:
                        # If no newline, it means the entire content is just the thought.
                        # In this case, we display nothing to the user for the thought.
                        display_content = "" # Or you could set it to a default "Thinking..." if desired.

                if display_content: # Only display if there's content after stripping thought
                    st.markdown(display_content)
                
                if last_ai_message.tool_calls:
                    st.info(f"Tool Call Initiated: {last_ai_message.tool_calls[0].name}")

            # Add the AI's response (or tool invocation message) to chat history
            st.session_state.chat_history.append(last_ai_message)

            # --- Handle MCQ Presentation ---
            # Only display MCQ if it's active, has a question, and hasn't been displayed yet in this run
            if final_state["mcq_active"] and final_state["mcq_question"] and not st.session_state.mcq_displayed:
                with st.chat_message("assistant"):
                    st.subheader("Multiple Choice Question:")
                    st.write(final_state["mcq_question"])
                    
                    mcq_options_dict = {chr(65 + i): option for i, option in enumerate(final_state["mcq_options"])}
                    selected_option = st.radio("Choose your answer:", list(mcq_options_dict.keys()), key="mcq_radio")

                    if st.button("Submit Answer", key="mcq_submit_button"):
                        st.session_state.mcq_displayed = True # Mark MCQ as displayed
                        user_answer_key = selected_option
                        correct_answer_key = final_state["mcq_correct_answer"]

                        is_correct = (user_answer_key == correct_answer_key)
                        
                        feedback_message = ""
                        if is_correct:
                            feedback_message = f"That's correct! The answer is {correct_answer_key}) {mcq_options_dict[correct_answer_key]}. Excellent!"
                            st.session_state.socratic_agent_state["user_struggle_count"] = 0
                            # Add logic to increase difficulty here if desired
                        else:
                            feedback_message = f"That's not quite right. The correct answer was {correct_answer_key}) {mcq_options_dict[correct_answer_key]}. Let's review this concept."
                            st.session_state.socratic_agent_state["user_struggle_count"] += 1
                            # Add logic to decrease difficulty or re-explain here

                        with st.chat_message("assistant"):
                            st.markdown(feedback_message)
                        st.session_state.chat_history.append(AIMessage(content=feedback_message))

                        # Reset MCQ state after submission
                        st.session_state.socratic_agent_state["mcq_active"] = False
                        st.session_state.socratic_agent_state["mcq_question"] = ""
                        st.session_state.socratic_agent_state["mcq_options"] = []
                        st.session_state.socratic_agent_state["mcq_correct_answer"] = ""
                        st.session_state.mcq_displayed = False # Reset for next MCQ
                        st.rerun() # Rerun to clear MCQ UI elements and continue chat
                # Set mcq_displayed to True after the radio button and submit button are rendered
                # to prevent re-rendering the MCQ in the same execution cycle if user_input is empty.
                # This ensures the MCQ only appears once per generation until submitted.
                st.session_state.mcq_displayed = True
        else:
            # If the LLM's response was only a tool call (no direct content),
            # the chat history will still show the tool invocation.
            pass

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Error during graph invocation: {e}", exc_info=True)

# Add a sidebar for potential future controls or information
st.sidebar.header("Tutor Settings")
st.sidebar.write(f"Current Difficulty: {st.session_state.socratic_agent_state['difficulty_level']}")
st.sidebar.write(f"Current Topic: {st.session_state.socratic_agent_state['topic']}")
st.sidebar.write(f"Current Sub-topic: {st.session_state.socratic_agent_state['sub_topic']}")
st.sidebar.write(f"Struggle Count: {st.session_state.socratic_agent_state['user_struggle_count']}")

# Optional: Button to clear chat history and reset state
if st.sidebar.button("Clear Chat and Reset"):
    st.session_state.chat_history = []
    st.session_state.socratic_agent_state = SocraticAgentState(
        messages=[],
        difficulty_level="begstreamlit run main.pyinner",
        user_struggle_count=0,
        topic="Python Basics",
        sub_topic="Introduction",
        mcq_active=False,
        mcq_question="",
        mcq_options=[],
        mcq_correct_answer="",
        agent_thought=""
    )
    st.session_state.initial_greeting_done = False
    st.session_state.mcq_displayed = False
    st.rerun()
