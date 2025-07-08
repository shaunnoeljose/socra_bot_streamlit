import streamlit as st
from your_module import SocraticAgentState, step_through_agent  # Replace with your actual module

# Initialize session state
if 'agent_state' not in st.session_state:
    st.session_state.agent_state = SocraticAgentState(
        messages=[],
        next_node='socratic_question',  # Or 'call_supervisor' for code input
        tool_input={}
    )

st.title("🧠 Socratic Python Tutor")

# Show conversation so far
st.markdown("### 🗨️ Conversation")
for msg in st.session_state.agent_state.messages:
    if hasattr(msg, 'content'):
        if msg.__class__.__name__ == 'HumanMessage':
            st.markdown(f"**👤 You:** {msg.content}")
        elif msg.__class__.__name__ == 'AIMessage':
            st.markdown(f"**🤖 Tutor:** {msg.content}")
        elif msg.__class__.__name__ == 'ToolMessage':
            st.markdown(f"**🛠️ Tool:** {msg.content}")

# Text input for user message
with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question, request help, or enter code:")
    submitted = st.form_submit_button("Send")

# On user input, update agent state
if submitted and user_input.strip():
    from langchain.schema import HumanMessage
    st.session_state.agent_state.messages.append(HumanMessage(content=user_input))

    # If needed, redirect based on type of input
    if user_input.strip().startswith("debug") or "print(" in user_input:
        st.session_state.agent_state.next_node = "call_supervisor"
    else:
        st.session_state.agent_state.next_node = "socratic_question"

    # Step agent
    st.session_state.agent_state = step_through_agent(st.session_state.agent_state)
    st.experimental_rerun()

# Show MCQ if available
if st.session_state.agent_state.mcq_active:
    st.markdown("### ❓ Multiple Choice Question")
    st.markdown(f"**{st.session_state.agent_state.mcq_question}**")
    for option in st.session_state.agent_state.mcq_options:
        st.markdown(f"- {option}")
    st.success(f"✅ Correct Answer: {st.session_state.agent_state.mcq_correct_answer}")
