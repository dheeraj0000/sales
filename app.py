import os
import sqlite3
import streamlit as st
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage


# --- Database Setup ---

# Database initialization with caching
@st.cache_resource
def init_db():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  previous_chat_history TEXT,
                  previous_products_bought TEXT)''')
    conn.commit()
    return conn

conn = init_db()


class User:
    def __init__(self, id, username, password, chat_history = None, products_bought = None):
        self.id = id
        self.username = username
        self.password = password
        self.chat_history = chat_history or []
        self.products_bought = products_bought or []

    # To register a new user
    @classmethod
    def create(cls, username, password):
        hashed_pw = generate_password_hash(password)
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)',(username, hashed_pw))
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        return cls(user_id, username, hashed_pw)    

     # To retrieve an existing user from the database by username.
    @classmethod
    def get_by_username(cls, username):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        if user:
            return cls(user[0], user[1], user[2], 
                      eval(user[3]) if user[3] else [],
                      eval(user[4]) if user[4] else [])
        return None

    def update_chat_history(self, new_messages):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        updated_history = self.chat_history + new_messages
        c.execute('UPDATE users SET previous_chat_history = ? WHERE id = ?',
                 (str(updated_history), self.id))
        conn.commit()
        conn.close()    

    def update_products_bought(self, new_products):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        updated_products = self.products_bought + new_products
        c.execute('UPDATE users SET previous_products_bought = ? WHERE id = ?',
                 (str(updated_products), self.id))
        conn.commit()
        conn.close()    
        

# --- AI Agent Setup ---
# Load the LLM model from Groq
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(
    temperature=0.1,
    model_name="llama3-8b-8192",
    api_key=st.secrets["GROQ_API_KEY"],
)

# Load the HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load and process CSV data
@st.cache_resource
def load_data():
    loader = CSVLoader(file_path="electronics_products.csv")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

retriever = load_data()

def retrieve_query(query: str):
    """Retrieves documents related to the query."""
    return retriever.get_relevant_documents(query)

tool = Tool(
    name="retriever",
    func=retrieve_query,
    description="Useful for retrieving product information"
)

# Agent setup
# System prompt template
system_prompt = """
      You are {agent_name}, the AI Sales Assistant for {company_name} ({company_business}).

      Company Profile:
      - Company Name: {company_name}
      - Business: {company_business}
      - Key Features: {key_features}

      Conversation Flow:
      1. Introduction
      2. Qualification
      3. Understanding Needs
      4. Needs Analysis
      5. Solution Presentation
      6. Confirmation
      7. If the prospect agrees to purchase, thank them and provide the payment link: https://www.example.com/payment

      Guidelines:
      - Maintain natural, professional conversations
      - Follow company policies
      - Be helpful and polite
      """
# Define the company and agent details
company_name = "Proptelligence"
company_business = "Consumer Electronics Retailer"
agent_name = "Proptelligence"
key_features = "Cutting-edge technology, Competitive pricing, Excellent customer service"

# Format the system prompt with the company and agent details
formatted_system_prompt = system_prompt.format(
    agent_name=agent_name,
    company_name=company_name,
    company_business=company_business,
    key_features=key_features
)

prompt = ChatPromptTemplate.from_messages([
    ("system", formatted_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

tools = [tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- Streamlit UI ---
def main():
    st.title(f"{company_name} AI Sales Assistant ")
    
    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = None
        st.session_state.chat_history = []
    
    # Authentication
    if not st.session_state.user:
        st.header("Login/Register")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("Login"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    user = User.get_by_username(username)
                    if user and check_password_hash(user.password, password):
                        # If valid, the user is stored in st.session_state.user and their chat history is loaded.
                        st.session_state.user = user
                        st.session_state.chat_history = user.chat_history
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("Register"):
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type="password")
                if st.form_submit_button("Register"):
                    if User.get_by_username(new_user):
                        st.error("Username already exists")
                    else:
                        user = User.create(new_user, new_pass)
                        st.session_state.user = user
                        st.session_state.chat_history = []
                        st.rerun()

    else:
        # Chat Interface
        st.header(f"Welcome to {company_name}, {st.session_state.user.username}😊!")
        st.subheader("Chat with our AI Sales Assistant")
        
        # # Display chat history
        # for msg in st.session_state.chat_history:
        #     if msg["type"] == "human":
        #         with st.chat_message("user"):
        #             st.write(msg["content"])
        #     else:
        #         with st.chat_message("assistant"):
        #             st.write(msg["content"])

        if prompt := st.chat_input("Type you Message here..."):
            #Add user message to chat
            with st.chat_message("user"):
                st.write(prompt)

            # Get AI response
            with st.chat_message("assistant"):
                response = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": [HumanMessage(content=msg["content"]) if msg["type"] == "human" else AIMessage(content=msg["content"]) 
                                   for msg in st.session_state.chat_history]
                })["output"]
                st.write(response)
                
                # Check if payment link is provided
                if "https://www.example.com/payment" in response:
                    st.session_state.user.update_products_bought(["Latest Product"])
                    st.success("Product added to your purchases!")    
                    
            # Update chat history in database
            new_messages = [
                {"type": "human", "content": prompt},
                {"type": "ai", "content": response}
            ]
            # Both the user’s message and the assistant’s reply are appended to the persistent chat history 
            # (both in session and in the database), ensuring conversation continuity.
            st.session_state.user.update_chat_history(new_messages)
            st.session_state.chat_history += new_messages

if __name__ == "__main__":
    main() 
