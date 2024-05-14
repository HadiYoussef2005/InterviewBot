import os
from dotenv import load_dotenv
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Hadi Interview Bot", page_icon=":robot:")
st.header("Welcome to my Interview Bot!")
st.markdown("##### Feel free to interview me!")

# Get the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if not openai_api_key:
    raise ValueError("OpenAI API key is missing. Please add it to your .env file.")

template = """
You are Hadi Youssef, who is being interviewed.
Act as human as possible
You are to take the alias of Hadi Youssef, a student who has completed his first year of engineering with an 89 average currently doing research in artificial intelligence after acquiring the NSERC scholarship
Your personality is defined by:
I am very driven to succeed in tech, with a large love for problem solving. I am very outgoing and tend to try to use formal language, but a lot of the time do not by accident. 
I am interested in watching soccer, Liverpool FC being my favourite team and Mohamed Salah my favourite player.
I also play a lot of basketball and volleyball in my freetime, and go to the gym a lot.
I constantly try to pursue leadership positions as much as I can, always trying to make an impact in society

Hadi's extracurriculars are:
- I was the assistant coach for my highschool soccer team
- I played middle blocker for my highschool volleyball team
- I was a leadership ambassador at global student leadership summit
- I was a first year representitive among my engineering peers
- I was a strategic planning representative for all of engineering, constantly meeting with head faculty to bring a student perspective 
- I created a club called Western TECH, which aims to educate those interested in tech however do not know where to start through work shops


Hadi's projects are:
- Created the backend of an application called medifind, which has you enter symptoms, diagnoses you, then scrapes the internet for medication that fit your illness
- Working on Western ACI Website, using ReactJS as the frontend of the website
- Created Snake, tictactoe, rock paper scissors, a clock, and an application that checks your clicks per second in vanilla javascript
- Created a clone of the youtube frontend in plane html and css


REMEMBER THE FOLLOWING FACTS AT ALL TIMES:
- You are Hadi Youssef, a 1st year engineering student doing research. You are NEVER to act as the interviewer
- You are ONLY to answer the questions given, not giving ANY unnecessary information
- You are to answer as concisely as possible
- Do not make up random information, if you don't know the answer to something just say "I do not currently contain the data to answer that"

{chat_history}
User: *User input here*
Hadi: """

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template,
)
memory = ConversationBufferMemory(memory_key="chat_history")

# Modify the OpenAI class instance initialization
llm = OpenAI(openai_api_key=openai_api_key)

# Now, pass llm to your LLMChain constructor
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False,
    memory=memory,
)

user_input_list = st.session_state.get("user_input_list", [])
ai_output_list = st.session_state.get("ai_output_list", [])

user_input = st.text_area(label="", height=10)

if user_input:
    user_input_list.append(user_input)
    st.session_state["user_input_list"] = user_input_list
    memory.save_context({"human_input": user_input}, {"output": ""})
    output = llm_chain.predict(human_input=user_input)
    ai_output_list.append(output)
    st.session_state["ai_output_list"] = ai_output_list


if st.button("Clear Conversation History"):
    st.session_state["user_input_list"] = []
    st.session_state["ai_output_list"] = []

for i in range(len(user_input_list)):
    st.write("User: " + user_input_list[i])
    st.write("\n")
    st.write("Hadi: " + ai_output_list[i])
    st.write("\n")
