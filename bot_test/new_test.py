# main.py

import streamlit as st
import os
import copy
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Mock the dependencies that aren't available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv():
        pass

# Mock LangChain message types
class BaseMessage:
    def __init__(self, content: str):
        self.content = content
        self.timestamp = datetime.now()

class HumanMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(content)

class AIMessage(BaseMessage):
    def __init__(self, content: str, tool_calls: Optional[List] = None):
        super().__init__(content)
        self.tool_calls = tool_calls or []

class ToolMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(content)

# Mock SocraticAgentState
@dataclass
class SocraticAgentState:
    messages: List[BaseMessage] = field(default_factory=list)
    difficulty_level: str = "beginner"
    user_struggle_count: int = 0
    topic: str = "Python Basics"
    sub_topic: str = "Introduction"
    mcq_active: bool = False
    mcq_question: str = ""
    mcq_options: List[str] = field(default_factory=list)
    mcq_correct_answer: str = ""
    agent_thought: str = ""

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)

# Mock memory saver
class MockMemorySaver:
    def __init__(self):
        self.storage = {}
    
    def get(self, config):
        thread_id = config.get("configurable", {}).get("thread_id")
        return self.storage.get(thread_id)
    
    def put(self, config, state):
        thread_id = config.get("configurable", {}).get("thread_id")
        self.storage[thread_id] = state

memory_saver = MockMemorySaver()

# Mock Socratic Graph
class MockSocraticGraph:
    def __init__(self):
        self.sample_mcqs = [
            {
                "question": "What is the correct way to define a function in Python?",
                "options": ["function myFunc():", "def myFunc():", "define myFunc():", "func myFunc():"],
                "correct_answer": "B"
            },
            {
                "question": "Which data type is used to store multiple items in Python?",
                "options": ["string", "integer", "list", "boolean"],
                "correct_answer": "C"
            },
            {
                "question": "How do you print 'Hello World' in Python?",
                "options": ["echo('Hello World')", "print('Hello World')", "console.log('Hello World')", "System.out.println('Hello World')"],
                "correct_answer": "B"
            }
        ]
        self.mcq_index = 0
    
    def invoke(self, state: SocraticAgentState, config: Dict[str, Any]) -> Dict[str, Any]:
        # Save current state
        memory_saver.put(config, state.__dict__)
        
        # Get the last message
        last_message = state.messages[-1] if state.messages else None
        
        if not last_message:
            return state.__dict__
        
        # Process the message
        if isinstance(last_message, HumanMessage):
            response = self._generate_response(last_message.content, state)
            state.messages.append(AIMessage(content=response))
            
            # Check if we should activate MCQ
            if any(keyword in last_message.content.lower() for keyword in ["mcq", "quiz", "test", "challenge"]):
                self._activate_mcq(state)
        
        return state.__dict__
    
    def _generate_response(self, user_input: str, state: SocraticAgentState) -> str:
        # Handle MCQ responses
        if "My answer to the MCQ is:" in user_input:
            return self._handle_mcq_response(user_input, state)
        
        # Generate contextual responses based on input
        user_input_lower = user_input.lower()
        
        if any(keyword in user_input_lower for keyword in ["function", "def", "method"]):
            state.topic = "Functions"
            state.sub_topic = "Function Definition"
            return """Great! Let's explore Python functions. 

Functions are reusable blocks of code that perform specific tasks. In Python, we define functions using the `def` keyword:

```python
def greet(name):
    return f"Hello, {name}!"
```

What would you like to know about functions? How to define them, call them, or perhaps about parameters and return values?"""
        
        elif any(keyword in user_input_lower for keyword in ["list", "array", "data structure"]):
            state.topic = "Data Structures"
            state.sub_topic = "Lists"
            return """Excellent choice! Lists are one of Python's most versatile data structures.

A list is an ordered collection of items that can be changed (mutable). Here's how you create one:

```python
my_list = [1, 2, 3, "hello", True]
```

Lists can contain different data types and you can:
- Access items by index: `my_list[0]`
- Add items: `my_list.append("new item")`
- Remove items: `my_list.remove("hello")`

What specific aspect of lists would you like to explore?"""
        
        elif any(keyword in user_input_lower for keyword in ["loop", "for", "while", "iterate"]):
            state.topic = "Control Flow"
            state.sub_topic = "Loops"
            return """Perfect! Loops are essential for repeating code efficiently.

Python has two main types of loops:

**For loops** - iterate over sequences:
```python
for item in [1, 2, 3]:
    print(item)
```

**While loops** - repeat while a condition is true:
```python
count = 0
while count < 3:
    print(count)
    count += 1
```

Which type of loop would you like to practice with?"""
        
        elif any(keyword in user_input_lower for keyword in ["variable", "assign", "value"]):
            state.topic = "Variables"
            state.sub_topic = "Assignment"
            return """Variables are containers for storing data values. In Python, you don't need to declare variable types explicitly:

```python
name = "Alice"        # String
age = 25             # Integer
height = 5.6         # Float
is_student = True    # Boolean
```

Python is dynamically typed, so you can reassign variables to different types:
```python
x = 10      # x is an integer
x = "hello" # now x is a string
```

What would you like to practice with variables?"""
        
        elif any(keyword in user_input_lower for keyword in ["mcq", "quiz", "test", "challenge"]):
            return "Great! I'll prepare a multiple-choice question for you to test your Python knowledge."
        
        else:
            return f"""I see you're interested in "{user_input}". 

As your Socratic tutor, I'd like to understand what you already know. Can you tell me:
- Have you worked with this concept before?
- What specific aspect would you like to learn or practice?

I can help you with:
- **Functions**: Creating reusable code blocks
- **Data Structures**: Lists, dictionaries, tuples
- **Control Flow**: If statements, loops
- **Variables**: Storing and manipulating data
- **MCQs**: Test your knowledge with questions

What interests you most?"""
    
    def _activate_mcq(self, state: SocraticAgentState):
        mcq = self.sample_mcqs[self.mcq_index % len(self.sample_mcqs)]
        state.mcq_active = True
        state.mcq_question = mcq["question"]
        state.mcq_options = mcq["options"]
        state.mcq_correct_answer = mcq["correct_answer"]
        self.mcq_index += 1
    
    def _handle_mcq_response(self, user_input: str, state: SocraticAgentState) -> str:
        # Extract the selected option (A, B, C, or D)
        selected = user_input.split(":")[-1].strip().replace(")", "")
        
        correct_answer = state.mcq_correct_answer
        
        if selected == correct_answer:
            state.user_struggle_count = max(0, state.user_struggle_count - 1)
            if state.difficulty_level == "beginner" and state.user_struggle_count == 0:
                state.difficulty_level = "intermediate"
            return f"""🎉 Excellent! You selected {selected}, which is correct!

{self._get_explanation(state.mcq_question, correct_answer, state.mcq_options)}

You're doing great! Would you like another challenge or shall we explore a different Python topic?"""
        else:
            state.user_struggle_count += 1
            if state.user_struggle_count >= 3 and state.difficulty_level == "intermediate":
                state.difficulty_level = "beginner"
            
            return f"""Not quite right. You selected {selected}, but the correct answer is {correct_answer}.

{self._get_explanation(state.mcq_question, correct_answer, state.mcq_options)}

Don't worry! Learning is a process. Would you like to try another question or review this concept more?"""
    
    def _get_explanation(self, question: str, correct_answer: str, options: List[str]) -> str:
        explanations = {
            "What is the correct way to define a function in Python?": "In Python, functions are defined using the `def` keyword followed by the function name and parentheses.",
            "Which data type is used to store multiple items in Python?": "Lists are ordered collections that can store multiple items of different data types.",
            "How do you print 'Hello World' in Python?": "The `print()` function is used to display output in Python."
        }
        return explanations.get(question, "Good question! This tests fundamental Python concepts.")

socratic_graph = MockSocraticGraph()

# Streamlit App Configuration
st.set_page_config(page_title="Socratic Python Tutor", page_icon="🐍")
st.title("🐍 Socratic Python Tutor")
st.markdown("---")

# Mock API key check
api_key_available = True  # Set to False to simulate missing API key
if not api_key_available:
    st.error("GOOGLE_API_KEY not found in environment. Please check your .env file.")
    st.stop()

# --- User ID Input (Sidebar) ---
st.sidebar.header("User Management")
user_id_input = st.sidebar.text_input("Enter your User ID:", value=st.session_state.get("user_id", "default_user"))

# Update session state with the current user ID
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
        agent_thought=""
    )
    st.session_state.initial_greeting_done = False
    st.session_state.mcq_input_displayed = False
    st.session_state.mcq_just_submitted = False
    st.rerun()

st.session_state.user_id = user_id_input

# --- Load State Button ---
if st.sidebar.button("💾 Load Saved State"):
    if st.session_state.user_id:
        try:
            loaded_state = memory_saver.get({"configurable": {"thread_id": st.session_state.user_id}})
            if loaded_state:
                st.session_state.socratic_agent_state = SocraticAgentState(**loaded_state)
                st.session_state.chat_history = loaded_state.get("messages", [])
                st.session_state.initial_greeting_done = True
                st.session_state.mcq_input_displayed = st.session_state.socratic_agent_state.mcq_active
                st.session_state.mcq_just_submitted = False
                st.sidebar.success(f"State loaded for User ID: '{st.session_state.user_id}'")
            else:
                st.sidebar.info(f"No saved state found for User ID: '{st.session_state.user_id}'. Starting new session.")
                st.session_state.chat_history = []
                st.session_state.socratic_agent_state = SocraticAgentState()
                st.session_state.initial_greeting_done = False
                st.session_state.mcq_input_displayed = False
                st.session_state.mcq_just_submitted = False
        except Exception as e:
            st.sidebar.error(f"Error loading state: {e}")
    else:
        st.sidebar.warning("Please enter a User ID to load a state.")
    st.rerun()

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "socratic_agent_state" not in st.session_state:
    st.session_state.socratic_agent_state = SocraticAgentState()

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
user_input = st.chat_input("Your message:", disabled=st.session_state.mcq_input_displayed)

# --- Handle User Input ---
if user_input and not st.session_state.mcq_input_displayed and not st.session_state.mcq_just_submitted:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    current_state = copy.deepcopy(st.session_state.socratic_agent_state)
    current_state.messages.append(HumanMessage(content=user_input))
    st.session_state.socratic_agent_state = current_state

    try:
        final_state = socratic_graph.invoke(
            st.session_state.socratic_agent_state,
            config={"configurable": {"thread_id": st.session_state.user_id}}
        )
        st.session_state.socratic_agent_state = SocraticAgentState(**final_state)

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
        
        if st.session_state.socratic_agent_state.mcq_active and not st.session_state.mcq_input_displayed:
            st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Display MCQ Widgets ---
if (st.session_state.socratic_agent_state.mcq_active and 
    st.session_state.socratic_agent_state.mcq_question and 
    not st.session_state.mcq_just_submitted):
    
    st.session_state.mcq_input_displayed = True

    options = st.session_state.socratic_agent_state.mcq_options
    option_map = {chr(65 + i): option for i, option in enumerate(options)}

    with st.form(key="mcq_form"):
        st.markdown(f"**{st.session_state.socratic_agent_state.mcq_question}**")
        selected_option_form = st.radio(
            "Choose your answer:",
            list(option_map.keys()),
            format_func=lambda x: f"{x}) {option_map[x]}",
            key=f"mcq_radio_form_{st.session_state.socratic_agent_state.mcq_question}",
            index=None
        )
        submit_button = st.form_submit_button("Submit Answer")

        if submit_button and selected_option_form is not None:
            mcq_response_message = f"My answer to the MCQ is: {selected_option_form})"
            st.session_state.chat_history.append(HumanMessage(content=mcq_response_message))
            
            current_state = copy.deepcopy(st.session_state.socratic_agent_state)
            current_state.messages.append(HumanMessage(content=mcq_response_message))
            st.session_state.socratic_agent_state = current_state

            st.session_state.mcq_just_submitted = True
            st.session_state.mcq_input_displayed = False
            st.rerun()
            
        elif submit_button and selected_option_form is None:
            st.warning("Please select an option before submitting.")

# --- Process MCQ Submission ---
if st.session_state.mcq_just_submitted:
    try:
        final_state = socratic_graph.invoke(
            st.session_state.socratic_agent_state,
            config={"configurable": {"thread_id": st.session_state.user_id}}
        )
        st.session_state.socratic_agent_state = SocraticAgentState(**final_state)

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
        
        st.session_state.socratic_agent_state.mcq_active = False
        st.session_state.mcq_just_submitted = False
        st.session_state.mcq_input_displayed = False
        st.rerun()

    except Exception as e:
        st.error(f"An error occurred during MCQ submission processing: {e}")
        st.session_state.mcq_just_submitted = False
        st.session_state.mcq_input_displayed = False
        st.session_state.socratic_agent_state.mcq_active = False
        st.rerun()

# --- Sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("Tutor Settings")
st.sidebar.write(f"Current User ID: `{st.session_state.get('user_id', 'None')}`")
st.sidebar.write(f"Current Difficulty: {st.session_state.socratic_agent_state.difficulty_level}")
st.sidebar.write(f"Topic: {st.session_state.socratic_agent_state.topic}")
st.sidebar.write(f"Sub-topic: {st.session_state.socratic_agent_state.sub_topic}")
st.sidebar.write(f"Struggle Count: {st.session_state.socratic_agent_state.user_struggle_count}")

if st.sidebar.button("🔄 Reset Chat (and current user state)"):
    st.session_state.chat_history = []
    st.session_state.socratic_agent_state = SocraticAgentState()
    st.session_state.initial_greeting_done = False
    st.session_state.mcq_input_displayed = False
    st.session_state.mcq_just_submitted = False
    
    if st.session_state.user_id:
        try:
            st.sidebar.info(f"State for User ID '{st.session_state.user_id}' will be reset on next interaction.")
        except Exception as e:
            st.sidebar.warning(f"Could not clear memory for User ID '{st.session_state.user_id}': {e}")
    st.rerun()
    
# --- Instructions ---
st.sidebar.markdown("---")
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. **Chat**: Ask about Python topics like functions, lists, loops, variables
2. **MCQ**: Type "quiz" or "mcq" to get test questions
3. **Save/Load**: Use User ID to save your progress
4. **Reset**: Clear all chat history and start fresh

**Try asking about:**
- Functions and methods
- Lists and data structures  
- Loops and control flow
- Variables and data types
- Request an MCQ quiz
""")