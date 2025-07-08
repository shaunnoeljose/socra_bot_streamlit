import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import List, Optional

# -----------------------------
# AGENT STATE CLASS
# -----------------------------
class SocraticAgentState:
    def __init__(self,
                 messages: Optional[List] = None,
                 next_node: str = "socratic_question",
                 tool_input: Optional[dict] = None,
                 difficulty_level: str = "beginner",
                 topic: str = "Functions",
                 sub_topic: str = "MCQ",
                 user_struggle_count: int = 0,
                 mcq_active: bool = False,
                 mcq_question: Optional[str] = None,
                 mcq_options: Optional[List[str]] = None,
                 mcq_correct_answer: Optional[str] = None,
                 agent_thought: Optional[str] = ""):

        self.messages = messages or []
        self.next_node = next_node
        self.tool_input = tool_input or {}
        self.difficulty_level = difficulty_level
        self.topic = topic
        self.sub_topic = sub_topic
        self.user_struggle_count = user_struggle_count
        self.mcq_active = mcq_active
        self.mcq_question = mcq_question
        self.mcq_options = mcq_options or []
        self.mcq_correct_answer = mcq_correct_answer
        self.agent_thought = agent_thought

# -----------------------------
# DUMMY NODE REGISTRY
# -----------------------------

def socratic_question_node(state: SocraticAgentState):
    user_msg = state.messages[-1].content if state.messages else ""
    response = AIMessage(content="In simple terms, what do you think a variable represents in programming?")
    state.messages.append(response)
    return state

def code_analysis_node(state: SocraticAgentState):
    tool_msg = ToolMessage(
        content="Code Analysis Result: Your code looks fine. It prints 'hello'.",
        name="code_analysis_agent",
        tool_call_id="001"
    )
    state.messages.append(tool_msg)
    return state

def generate_mcq_node(state: SocraticAgentState):
    state.mcq_active = True
    state.mcq_question = "Which of the following operations would lead to an `IndentationError` in Python?"
    state.mcq_options = [
        "A) Missing a colon after a function definition",
        "B) Inconsistent use of spaces and tabs for indentation",
        "C) Using a reserved keyword as a variable name",
        "D) Forgetting a closing parenthesis"
    ]
    state.mcq_correct_answer = "B"
    return state

node_registry = {
    "socratic_question": socratic_question_node,
    "code_analysis": code_analysis_node,
    "generate_mcq": generate_mcq_node
}

# -----------------------------
# AGENT STEP FUNCTION
# -----------------------------
def step_through_agent(state: SocraticAgentState):
    node_fn = node_registry.get(state.next_node)
    if node_fn:
        return node_fn(state)
    return state

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Socratic Python Tutor", page_icon="📘")

if "agent_state" not in st.session_state:
    st.session_state.agent_state = SocraticAgentState(
        messages=[],
        next_node="socratic_question",
        tool_input={}
    )

st.title("🧠 Socratic Python Tutor")

# Display conversation
st.markdown("### 🗨️ Conversation")
for msg in st.session_state.agent_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**👤 You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**🤖 Tutor:** {msg.content}")
    elif isinstance(msg, ToolMessage):
        st.markdown(f"**🛠️ Tool:** {msg.content}")

# User Input
with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question, request help, or enter code:")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.agent_state.messages.append(HumanMessage(content=user_input))
    if "print(" in user_input or user_input.strip().startswith("debug"):
        st.session_state.agent_state.next_node = "code_analysis"
    elif "MCQ" in user_input:
        st.session_state.agent_state.next_node = "generate_mcq"
    else:
        st.session_state.agent_state.next_node = "socratic_question"

    # Set a flag for rerun instead of calling rerun immediately
    st.session_state.run_agent = True

# Run the agent step only after form submission
if st.session_state.get("run_agent"):
    st.session_state.agent_state = step_through_agent(st.session_state.agent_state)
    st.session_state.run_agent = False
    st.experimental_rerun()

if st.session_state.agent_state.mcq_active:
    st.markdown("### ❓ Multiple Choice Question")
    st.markdown(f"**{st.session_state.agent_state.mcq_question}**")
    for option in st.session_state.agent_state.mcq_options:
        st.markdown(f"- {option}")
    st.success(f"✅ Correct Answer: {st.session_state.agent_state.mcq_correct_answer}")
