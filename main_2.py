# main.py

import streamlit as st
import os
import copy
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from socrabot_logic_2 import socratic_graph, SocraticAgentState

load_dotenv()

st.set_page_config(page_title="Socratic Python Tutor", page_icon="🐍")
st.title("🐍 Socratic Python Tutor")
st.markdown("---")

# Check if API key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in environment. Please check your .env file.")
    st.stop()

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "socratic_agent_state" not in st.session_state:
    st.session_state.socratic_agent_state = SocraticAgentState(
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

if "initial_greeting_done" not in st.session_state:
    st.session_state.initial_greeting_done = False

# This flag controls if the MCQ input widgets are currently displayed
if "mcq_input_displayed" not in st.session_state:
    st.session_state.mcq_input_displayed = False

# Flag to prevent re-processing graph if MCQ was just submitted
if "mcq_just_submitted" not in st.session_state:
    st.session_state.mcq_just_submitted = False

# --- Display Chat History ---
for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        if isinstance(message, ToolMessage):
            # ToolMessages are internal, don't display directly in chat bubbles
            st.info(f"Tool Output (Internal): {message.content}")
        else:
            display_content = message.content
            if isinstance(message, AIMessage) and display_content and display_content.startswith("Thought:"):
                parts = display_content.split("\n", 1)
                if len(parts) > 1:
                    display_content = parts[1].strip()
                else:
                    display_content = display_content.replace("Thought:", "").strip()
            
            if display_content:
                st.markdown(display_content)
            
            if isinstance(message, AIMessage) and message.tool_calls:
                 st.info(f"Agent called tool(s): {[tc['name'] for tc in message.tool_calls]}")


# --- Initial Greeting ---
if not st.session_state.initial_greeting_done:
    greeting = "Hello! I'm your Socratic Python Tutor. What Python topic would you like to learn or practice today? Or would you like to test your knowledge with a challenge or an MCQ?"
    with st.chat_message("assistant"):
        st.markdown(greeting)
    st.session_state.chat_history.append(AIMessage(content=greeting))
    st.session_state.initial_greeting_done = True

# --- Main Chat Input ---
user_input = st.chat_input("Your message:", disabled=st.session_state.mcq_input_displayed)

# --- Handle User Input or Process Graph ---
# Only process user_input if no MCQ is active and no MCQ was just submitted
if user_input and not st.session_state.mcq_input_displayed and not st.session_state.mcq_just_submitted: 
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    current_state = copy.deepcopy(st.session_state.socratic_agent_state)
    current_state["messages"].append(HumanMessage(content=user_input))
    st.session_state.socratic_agent_state = current_state # Update session state immediately

    try:
        final_state = socratic_graph.invoke(st.session_state.socratic_agent_state)
        st.session_state.socratic_agent_state = final_state

        last_ai_message = next(
            (msg for msg in reversed(final_state["messages"]) if isinstance(msg, AIMessage)), None
        )

        if last_ai_message:
            with st.chat_message("assistant"):
                content = last_ai_message.content
                if content and content.startswith("Thought:"):
                    parts = content.split("\n", 1)
                    if len(parts) > 1:
                        content = parts[1].strip()
                    else:
                        content = content.replace("Thought:", "").strip()
                if content:
                    st.markdown(content)
            st.session_state.chat_history.append(last_ai_message)
        
        # If an MCQ was just activated by the graph, trigger rerun to display it
        # This rerun will then hit the MCQ display block below
        if st.session_state.socratic_agent_state["mcq_active"] and not st.session_state.mcq_input_displayed:
            st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Display MCQ Widgets (outside chat bubble) ---
# Only display if MCQ is active AND we are not currently processing a submission
# and the MCQ question is actually set
if (
    st.session_state.socratic_agent_state["mcq_active"]
    and st.session_state.socratic_agent_state["mcq_question"]
    and not st.session_state.mcq_just_submitted 
):
    st.session_state.mcq_input_displayed = True # Set flag when widgets are shown

    options = st.session_state.socratic_agent_state["mcq_options"]
    option_map = {chr(65 + i): option for i, option in enumerate(options)}

    # Use a form to group the radio button and submit button
    # This prevents the radio button selection from triggering a rerun until the form is submitted
    with st.form(key="mcq_form"):
        # Only ONE st.radio here
        selected_option_form = st.radio(
            "Choose your answer:",
            list(option_map.keys()),
            key=f"mcq_radio_form_{st.session_state.socratic_agent_state['mcq_question']}",
            index=None # Allow no selection initially
        )
        submit_button = st.form_submit_button("Submit Answer")

        if submit_button and selected_option_form is not None:
            user_answer_chosen = option_map[selected_option_form]
            
            # Construct a message for the LLM to process the MCQ answer
            mcq_response_message = f"My answer to the MCQ is: {selected_option_form}) {user_answer_chosen}"
            
            # Add user's MCQ answer to chat history
            st.session_state.chat_history.append(HumanMessage(content=mcq_response_message))
            
            # Prepare state for graph invocation with user's MCQ answer
            current_state = copy.deepcopy(st.session_state.socratic_agent_state)
            current_state["messages"].append(HumanMessage(content=mcq_response_message))
            st.session_state.socratic_agent_state = current_state # Update session state

            # Set a flag to indicate that an MCQ answer has been submitted
            st.session_state.mcq_just_submitted = True
            st.session_state.mcq_input_displayed = False # Hide MCQ widgets immediately

            st.rerun() # Rerun to process the submitted answer through the graph
        elif submit_button and selected_option_form is None:
            st.warning("Please select an option before submitting.")


# --- Process MCQ Submission After Rerun ---
# This block runs on the rerun triggered by `st.form_submit_button`
# It will only execute if mcq_just_submitted is True
if st.session_state.mcq_just_submitted:
    try:
        final_state = socratic_graph.invoke(st.session_state.socratic_agent_state)
        st.session_state.socratic_agent_state = final_state

        last_ai_message = next(
            (msg for msg in reversed(final_state["messages"]) if isinstance(msg, AIMessage)), None
        )

        if last_ai_message:
            with st.chat_message("assistant"):
                content = last_ai_message.content
                if content and content.startswith("Thought:"):
                    parts = content.split("\n", 1)
                    if len(parts) > 1:
                        content = parts[1].strip()
                    else:
                        content = content.replace("Thought:", "").strip()
                if content:
                    st.markdown(content)
            st.session_state.chat_history.append(last_ai_message)
        
        # After processing and displaying the LLM's response,
        # explicitly reset mcq_active in the session state to hide widgets.
        # This is the key fix for the widgets remaining.
        st.session_state.socratic_agent_state["mcq_active"] = False 
        st.session_state.mcq_just_submitted = False 
        st.session_state.mcq_input_displayed = False 

        st.rerun() # Final rerun to display the LLM's response and ensure widgets are gone

    except Exception as e:
        st.error(f"An error occurred during MCQ submission processing: {e}")
        st.session_state.mcq_just_submitted = False # Ensure flag is reset on error
        st.session_state.mcq_input_displayed = False # Ensure widgets are hidden on error
        st.session_state.socratic_agent_state["mcq_active"] = False # Also reset on error
        st.rerun()


# --- Sidebar ---
st.sidebar.header("Tutor Settings")
st.sidebar.write(f"Current Difficulty: {st.session_state.socratic_agent_state['difficulty_level']}")
st.sidebar.write(f"Topic: {st.session_state.socratic_agent_state['topic']}")
st.sidebar.write(f"Sub-topic: {st.session_state.socratic_agent_state['sub_topic']}")
st.sidebar.write(f"Struggle Count: {st.session_state.socratic_agent_state['user_struggle_count']}")

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.socratic_agent_state = SocraticAgentState(
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
    st.session_state.initial_greeting_done = False
    st.session_state.mcq_input_displayed = False
    st.session_state.mcq_just_submitted = False
    st.rerun()
