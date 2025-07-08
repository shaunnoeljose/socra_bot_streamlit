# socratic_app.py

import streamlit as st
from socratic_bot_2 import socratic_graph, SocraticAgentState
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Socratic Bot", layout="wide")
st.title("🧠 Socratic Python Tutor")

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "bot_state" not in st.session_state:
    st.session_state.bot_state = SocraticAgentState(
        messages=[],
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

# Display chat history
for msg in st.session_state.chat_history:
    role = "👤 You" if msg.type == "human" else "🤖 Bot"
    with st.chat_message(role):
        st.markdown(msg.content)

# User input
user_input = st.chat_input("Ask a Python question or say 'give me an MCQ'...")

if user_input:
    # Show user message
    st.chat_message("👤 You").markdown(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.bot_state["messages"].append(HumanMessage(content=user_input))

    # Run the graph
    new_state = socratic_graph.invoke(st.session_state.bot_state)

    # Update state
    for msg in new_state["messages"]:
        st.session_state.chat_history.append(msg)
        st.session_state.bot_state["messages"].append(msg)

    st.session_state.bot_state.update(new_state)

    # Display bot message
    bot_msg = new_state["messages"][-1]
    with st.chat_message("🤖 Bot"):
        if st.session_state.bot_state["mcq_active"]:
            st.markdown(f"**{st.session_state.bot_state['mcq_question']}**")
            options = st.session_state.bot_state["mcq_options"]
            user_answer = st.radio("Choose your answer:", options, key=str(len(st.session_state.chat_history)))

            if st.button("Submit Answer", key=f"submit_{len(st.session_state.chat_history)}"):
                correct = st.session_state.bot_state["mcq_correct_answer"]
                if user_answer.startswith(correct):
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. The correct answer is: {correct}")
                st.session_state.bot_state["mcq_active"] = False  # Reset for next turn
        else:
            st.markdown(bot_msg.content)
