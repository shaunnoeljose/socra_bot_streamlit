# main.py

import streamlit as st
import os
import copy
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from sb_logic import enhanced_socratic_graph as socratic_graph, SocraticAgentState, memory_saver 

load_dotenv()

st.set_page_config(page_title="Socratic Python Tutor", page_icon="�")
st.title("💡 Socratic Python Tutor")
st.markdown("---")

# Check if API key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in environment. Please check your .env file.")
    st.stop()

# --- Initialize Session State ---
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"
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
if "mcq_input_displayed" not in st.session_state:
    st.session_state.mcq_input_displayed = False
if "mcq_just_submitted" not in st.session_state:
    st.session_state.mcq_just_submitted = False
if "show_user_id_popup" not in st.session_state:
    # Show popup if user_id is default or not yet set by user
    st.session_state.show_user_id_popup = (st.session_state.user_id == "default_user")


# --- User ID Input Popup ---
user_id_popup_placeholder = st.empty()

if st.session_state.show_user_id_popup:
    with user_id_popup_placeholder.container():
        st.info("Please enter a User ID to save your progress. You can use 'default_user' for a temporary session.")
        with st.form(key="user_id_form", clear_on_submit=False):
            new_user_id = st.text_input("Enter your User ID:", value=st.session_state.user_id, key="popup_user_id_input")
            submit_user_id = st.form_submit_button("Start Tutoring")

            if submit_user_id:
                if new_user_id.strip(): # Ensure the input is not just whitespace
                    if new_user_id.strip() != st.session_state.user_id:
                        st.session_state.user_id = new_user_id.strip()
                        # Clear state if user ID changes
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
                        st.session_state.show_user_id_popup = False # Hide popup
                        st.rerun()
                    else: # User submitted the same ID, just hide popup
                        st.session_state.show_user_id_popup = False
                        st.rerun()
                else:
                    st.warning("User ID cannot be empty. Please enter a valid ID.")
else:
    user_id_popup_placeholder.empty()


# --- Load State Button (remains in sidebar) ---
st.sidebar.header("User Management")
if st.sidebar.button("💾 Load Saved State"):
    if st.session_state.user_id and st.session_state.user_id != "default_user": # Don't load for default user
        try:
            loaded_state = memory_saver.get({"configurable": {"thread_id": st.session_state.user_id}})
            if loaded_state:
                st.session_state.socratic_agent_state = SocraticAgentState(**loaded_state)
                st.session_state.chat_history = loaded_state.get("messages", [])
                st.session_state.initial_greeting_done = True 
                st.session_state.mcq_input_displayed = st.session_state.socratic_agent_state["mcq_active"]
                st.session_state.mcq_just_submitted = False
                st.sidebar.success(f"State loaded for User ID: '{st.session_state.user_id}'")
            else:
                st.sidebar.info(f"No saved state found for User ID: '{st.session_state.user_id}'. Starting new session.")
                st.session_state.chat_history = []
                st.session_state.socratic_agent_state = SocraticAgentState(
                    messages=[], difficulty_level="beginner", user_struggle_count=0,
                    topic="Python Basics", sub_topic="Introduction", mcq_active=False,
                    mcq_question="", mcq_options=[], mcq_correct_answer="", agent_thought=""
                )
                st.session_state.initial_greeting_done = False
                st.session_state.mcq_input_displayed = False
                st.session_state.mcq_just_submitted = False
        except Exception as e:
            st.sidebar.error(f"Error loading state: {e}")
    else:
        st.sidebar.warning("Please enter a specific User ID (not 'default_user') to load a state.")
    st.rerun() 


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
# Only display greeting if popup is not active and greeting not done
if not st.session_state.initial_greeting_done and not st.session_state.show_user_id_popup:
    greeting = "Hello! I'm your Socratic Python Tutor. What Python topic would you like to learn or practice today? Or would you like to test your knowledge with a challenge or an MCQ?"
    with st.chat_message("assistant"):
        st.markdown(greeting)
    st.session_state.chat_history.append(AIMessage(content=greeting))
    st.session_state.initial_greeting_done = True

# --- Main Chat Input ---
# Disable chat input if popup is active
user_input = st.chat_input("Your message:", disabled=st.session_state.mcq_input_displayed or st.session_state.show_user_id_popup)

# --- Handle User Input or Process Graph ---
if user_input and not st.session_state.mcq_input_displayed and not st.session_state.mcq_just_submitted and not st.session_state.show_user_id_popup: 
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    current_state = copy.deepcopy(st.session_state.socratic_agent_state)
    current_state["messages"].append(HumanMessage(content=user_input))
    st.session_state.socratic_agent_state = current_state 

    try:
        thread_id_to_use = st.session_state.user_id if st.session_state.user_id else "default_user"
        prev_chat_history_len = len(st.session_state.chat_history) # Store current length
        
        final_state = socratic_graph.invoke(
            st.session_state.socratic_agent_state,
            config={"configurable": {"thread_id": thread_id_to_use}}
        )
        st.session_state.socratic_agent_state = SocraticAgentState(**final_state) 

        # Iterate through all new messages generated by the graph
        new_messages_from_graph = final_state["messages"][prev_chat_history_len:]
        
        for msg in new_messages_from_graph:
            if isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    content = msg.content
                    if content and content.startswith("Thought:"):
                        parts = content.split("\n", 1)
                        if len(parts) > 1:
                            content = parts[1].strip()
                        else:
                            content = content.replace("Thought:", "").strip()
                    if content:
                        st.markdown(content)
                st.session_state.chat_history.append(msg)
            elif isinstance(msg, ToolMessage):
                # ToolMessages are internal, just add to history
                st.session_state.chat_history.append(msg)
        
        if st.session_state.socratic_agent_state["mcq_active"] and not st.session_state.mcq_input_displayed:
            st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Display MCQ Widgets (outside chat bubble) ---
if (
    st.session_state.socratic_agent_state["mcq_active"]
    and st.session_state.socratic_agent_state["mcq_question"]
    and not st.session_state.mcq_just_submitted 
    and not st.session_state.show_user_id_popup 
):
    st.session_state.mcq_input_displayed = True 

    options = st.session_state.socratic_agent_state["mcq_options"]
    option_map = {chr(65 + i): option for i, option in enumerate(options)}

    with st.form(key="mcq_form"):
        st.markdown(f"**{st.session_state.socratic_agent_state['mcq_question']}**") 
        selected_option_form = st.radio(
            "Choose your answer:",
            list(option_map.keys()),
            key=f"mcq_radio_form_{st.session_state.socratic_agent_state['mcq_question']}",
            index=None 
        )
        submit_button = st.form_submit_button("Submit Answer")

        if submit_button and selected_option_form is not None:
            mcq_response_message = f"My answer to the MCQ is: {selected_option_form})" 
            
            st.session_state.chat_history.append(HumanMessage(content=mcq_response_message))
            
            current_state = copy.deepcopy(st.session_state.socratic_agent_state)
            current_state["messages"].append(HumanMessage(content=mcq_response_message))
            st.session_state.socratic_agent_state = current_state

            st.session_state.mcq_just_submitted = True
            st.session_state.mcq_input_displayed = False

            # No st.rerun() here, let the next block handle the invocation and display

        elif submit_button and selected_option_form is None:
            st.warning("Please select an option before submitting.")


# --- Process MCQ Submission After Rerun ---
# This block will now always run after an MCQ submission attempt
if st.session_state.mcq_just_submitted:
    try:
        thread_id_to_use = st.session_state.user_id if st.session_state.user_id else "default_user"
        prev_chat_history_len = len(st.session_state.chat_history) # Store current length
        
        final_state = socratic_graph.invoke(
            st.session_state.socratic_agent_state,
            config={"configurable": {"thread_id": thread_id_to_use}}
        )
        st.session_state.socratic_agent_state = SocraticAgentState(**final_state) 

        # Iterate through all new messages generated by the graph
        new_messages_from_graph = final_state["messages"][prev_chat_history_len:]
        
        for msg in new_messages_from_graph:
            if isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    content = msg.content
                    if content and content.startswith("Thought:"):
                        parts = content.split("\n", 1)
                        if len(parts) > 1:
                            content = parts[1].strip()
                        else:
                            content = content.replace("Thought:", "").strip()
                    if content:
                        st.markdown(content)
                st.session_state.chat_history.append(msg)
            elif isinstance(msg, ToolMessage):
                # ToolMessages are internal, just add to history
                st.session_state.chat_history.append(msg)
        
        st.session_state.socratic_agent_state["mcq_active"] = False 
        st.session_state.mcq_just_submitted = False 
        st.session_state.mcq_input_displayed = False 

        st.rerun() # Rerun to clear the MCQ form and update chat correctly

    except Exception as e:
        st.error(f"An error occurred during MCQ submission processing: {e}")
        st.session_state.mcq_just_submitted = False 
        st.session_state.mcq_input_displayed = False 
        st.session_state.socratic_agent_state["mcq_active"] = False 
        st.rerun()


# --- Sidebar (Tutor Settings and Reset) ---
st.sidebar.markdown("---") 
st.sidebar.header("Tutor Settings")
st.sidebar.write(f"Current User ID: `{st.session_state.get('user_id', 'None')}`")
st.sidebar.write(f"Current Difficulty: {st.session_state.socratic_agent_state['difficulty_level']}")
st.sidebar.write(f"Topic: {st.session_state.socratic_agent_state['topic']}")
st.sidebar.write(f"Sub-topic: {st.session_state.socratic_agent_state['sub_topic']}")
st.sidebar.write(f"Struggle Count: {st.session_state.socratic_agent_state['user_struggle_count']}")

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
        agent_thought=""
    )
    st.session_state.initial_greeting_done = False
    st.session_state.mcq_input_displayed = False
    st.session_state.mcq_just_submitted = False
    st.session_state.show_user_id_popup = (st.session_state.user_id == "default_user") 

    if st.session_state.user_id and st.session_state.user_id != "default_user":
        try:
            st.sidebar.info(f"State for User ID '{st.session_state.user_id}' will be reset on next interaction.")
        except Exception as e:
            st.sidebar.warning(f"Could not clear memory for User ID '{st.session_state.user_id}': {e}")
    else:
        st.sidebar.info("State for 'default_user' will be reset on next interaction.")
    st.rerun()

    