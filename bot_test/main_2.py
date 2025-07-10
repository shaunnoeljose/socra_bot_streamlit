# main.py

import streamlit as st
import os
import copy
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from sb_logic import socratic_graph, SocraticAgentState, memory_saver # Import memory_saver

load_dotenv()

st.set_page_config(page_title="Socratic Python Tutor", page_icon="🐍")
st.title("🐍 Socratic Python Tutor")
st.markdown("---")

# Check if API key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in environment. Please check your .env file.")
    st.stop()

# --- User ID Input (Sidebar) ---
st.sidebar.header("User Management")
user_id_input = st.sidebar.text_input("Enter your User ID:", value=st.session_state.get("user_id", "default_user"))

# Update session state with the current user ID
# This block should be carefully managed to avoid infinite reruns on ID change
if user_id_input != st.session_state.get("user_id"):
    st.session_state.user_id = user_id_input
    # Clear chat history and current agent state when user ID changes
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
        mcq_explanation="", # Added missing key from TypedDict
        agent_thought="",
        next_node_decision="" # Added missing key from TypedDict
    )
    st.session_state.initial_greeting_done = False
    st.session_state.mcq_input_displayed = False
    st.session_state.mcq_just_submitted = False
    st.rerun() 

st.session_state.user_id = user_id_input # Ensure user_id is always up-to-date in session_state

# --- Load State Button ---
if st.sidebar.button("💾 Load Saved State"):
    if st.session_state.user_id:
        try:
            # Load the state from MemorySaver using the current user_id
            loaded_state = memory_saver.get({"configurable": {"thread_id": st.session_state.user_id}})
            if loaded_state:
                # Ensure all SocraticAgentState keys are present, providing defaults if not
                default_state = SocraticAgentState(
                    messages=[], difficulty_level="beginner", user_struggle_count=0,
                    topic="", sub_topic="", mcq_active=False, mcq_question="",
                    mcq_options=[], mcq_correct_answer="", mcq_explanation="",
                    agent_thought="", next_node_decision=""
                )
                # Merge loaded state with defaults to ensure all keys exist
                for key, default_value in default_state.items():
                    if key not in loaded_state:
                        loaded_state[key] = default_value
                
                st.session_state.socratic_agent_state = SocraticAgentState(**loaded_state)
                st.session_state.chat_history = loaded_state.get("messages", [])
                st.session_state.initial_greeting_done = True 
                st.session_state.mcq_input_displayed = st.session_state.socratic_agent_state["mcq_active"]
                st.session_state.mcq_just_submitted = False
                st.sidebar.success(f"State loaded for User ID: '{st.session_state.user_id}'")
            else:
                st.sidebar.info(f"No saved state found for User ID: '{st.session_state.user_id}'. Starting new session.")
                # Reset to initial blank state if no saved state
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
                    mcq_explanation="", # Added missing key
                    agent_thought="",
                    next_node_decision="" # Added missing key
                )
                st.session_state.initial_greeting_done = False
                st.session_state.mcq_input_displayed = False
                st.session_state.mcq_just_submitted = False
        except Exception as e:
            st.sidebar.error(f"Error loading state: {e}")
    else:
        st.sidebar.warning("Please enter a User ID to load a state.")
    st.rerun() 

# --- Initialize Session State (if not loaded by user ID or first run) ---
# Ensure these initializations happen AFTER the potential load from memory_saver
# if the user_id block or load button haven't already handled it.
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
        mcq_explanation="", # Added missing key
        agent_thought="",
        next_node_decision="" # Added missing key
    )

if "initial_greeting_done" not in st.session_state:
    st.session_state.initial_greeting_done = False

if "mcq_input_displayed" not in st.session_state:
    st.session_state.mcq_input_displayed = False

if "mcq_just_submitted" not in st.session_state:
    st.session_state.mcq_just_submitted = False


# --- Display Chat History ---
for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        if isinstance(message, ToolMessage):
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
user_input = st.chat_input("Your message:", disabled=st.session_state.get("mcq_input_displayed", False)) # Use .get()

# --- Handle User Input or Process Graph ---
if user_input and not st.session_state.get("mcq_input_displayed", False) and not st.session_state.get("mcq_just_submitted", False): # Use .get()
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    current_state = copy.deepcopy(st.session_state.socratic_agent_state)
    current_state["messages"].append(HumanMessage(content=user_input))
    st.session_state.socratic_agent_state = current_state 

    try:
        final_state = socratic_graph.invoke(
            st.session_state.socratic_agent_state,
            config={"configurable": {"thread_id": st.session_state.user_id}}
        )
        # Safely update socratic_agent_state by merging with a default
        updated_socratic_state = SocraticAgentState(
            messages=[], difficulty_level="beginner", user_struggle_count=0,
            topic="", sub_topic="", mcq_active=False, mcq_question="",
            mcq_options=[], mcq_correct_answer="", mcq_explanation="",
            agent_thought="", next_node_decision=""
        )
        updated_socratic_state.update(final_state) # Update with graph's output
        st.session_state.socratic_agent_state = updated_socratic_state

        last_ai_message = next(
            (msg for msg in reversed(st.session_state.socratic_agent_state["messages"]) if isinstance(msg, AIMessage)), None # Changed from final_state to st.session_state.socratic_agent_state
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
        
        # Access mcq_active safely using .get()
        if st.session_state.socratic_agent_state.get("mcq_active", False) and not st.session_state.get("mcq_input_displayed", False):
            st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Display MCQ Widgets (outside chat bubble) ---
# Access mcq_active, mcq_question, mcq_just_submitted safely using .get()
if (
    st.session_state.socratic_agent_state.get("mcq_active", False)
    and st.session_state.socratic_agent_state.get("mcq_question", "")
    and not st.session_state.get("mcq_just_submitted", False)
):
    st.session_state.mcq_input_displayed = True 

    options = st.session_state.socratic_agent_state.get("mcq_options", []) # Use .get()
    option_map = {chr(65 + i): option for i, option in enumerate(options)}

    with st.form(key="mcq_form"):
        st.markdown(f"**{st.session_state.socratic_agent_state.get('mcq_question', 'MCQ Question')}**") # Use .get()
        selected_option_form = st.radio(
            "Choose your answer:",
            list(option_map.keys()),
            key=f"mcq_radio_form_{st.session_state.socratic_agent_state.get('mcq_question', time.time())}", # Use time.time() for unique key if question is empty
            index=None 
        )
        submit_button = st.form_submit_button("Submit Answer")

        if submit_button and selected_option_form is not None:
            user_answer_chosen = option_map[selected_option_form]
            
            mcq_response_message = f"My answer to the MCQ is: {selected_option_form})" 
            
            st.session_state.chat_history.append(HumanMessage(content=mcq_response_message))
            
            current_state = copy.deepcopy(st.session_state.socratic_agent_state)
            current_state["messages"].append(HumanMessage(content=mcq_response_message))
            st.session_state.socratic_agent_state = current_state

            st.session_state.mcq_just_submitted = True
            st.session_state.mcq_input_displayed = False

            st.rerun()
        elif submit_button and selected_option_form is None:
            st.warning("Please select an option before submitting.")


# --- Process MCQ Submission After Rerun ---
if st.session_state.get("mcq_just_submitted", False): # Use .get()
    try:
        final_state = socratic_graph.invoke(
            st.session_state.socratic_agent_state,
            config={"configurable": {"thread_id": st.session_state.user_id}}
        )
        # Safely update socratic_agent_state by merging with a default
        updated_socratic_state = SocraticAgentState(
            messages=[], difficulty_level="beginner", user_struggle_count=0,
            topic="", sub_topic="", mcq_active=False, mcq_question="",
            mcq_options=[], mcq_correct_answer="", mcq_explanation="",
            agent_thought="", next_node_decision=""
        )
        updated_socratic_state.update(final_state) # Update with graph's output
        st.session_state.socratic_agent_state = updated_socratic_state

        last_ai_message = next(
            (msg for msg in reversed(st.session_state.socratic_agent_state["messages"]) if isinstance(msg, AIMessage)), None # Changed from final_state to st.session_state.socratic_agent_state
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
        
        st.session_state.socratic_agent_state["mcq_active"] = False 
        st.session_state.mcq_just_submitted = False 
        st.session_state.mcq_input_displayed = False 

        st.rerun()

    except Exception as e:
        st.error(f"An error occurred during MCQ submission processing: {e}")
        st.session_state.mcq_just_submitted = False 
        st.session_state.mcq_input_displayed = False 
        st.session_state.socratic_agent_state["mcq_active"] = False 
        st.rerun()


# --- Sidebar ---
st.sidebar.markdown("---") 
st.sidebar.header("Tutor Settings")
st.sidebar.write(f"Current User ID: `{st.session_state.get('user_id', 'None')}`")
st.sidebar.write(f"Current Difficulty: {st.session_state.socratic_agent_state.get('difficulty_level', 'N/A')}") # Use .get()
st.sidebar.write(f"Topic: {st.session_state.socratic_agent_state.get('topic', 'N/A')}") # Use .get()
st.sidebar.write(f"Sub-topic: {st.session_state.socratic_agent_state.get('sub_topic', 'N/A')}") # Use .get()
st.sidebar.write(f"Struggle Count: {st.session_state.socratic_agent_state.get('user_struggle_count', 'N/A')}") # Use .get()

if st.sidebar.button("🔄 Reset Chat (and current user state)"):
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
        mcq_explanation="", # Added missing key
        agent_thought="",
        next_node_decision="" # Added missing key
    )
    st.session_state.initial_greeting_done = False
    st.session_state.mcq_input_displayed = False
    st.session_state.mcq_just_submitted = False
    
    if st.session_state.user_id:
        st.sidebar.info(f"State for User ID '{st.session_state.user_id}' will be reset on next interaction.")
    st.rerun()