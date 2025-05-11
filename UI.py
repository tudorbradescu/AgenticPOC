import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# Sidebar Inputs
st.sidebar.title("Client Engagement Inputs")
client_name   = st.sidebar.text_input("Client Name", help="Must match history_{client_name}.txt file")
person_name   = st.sidebar.text_input("Person Name", help="Must match persona_{person_name}.md file")
project_brief = st.sidebar.text_area("Project Brief / Needs")
run_button    = st.sidebar.button("Generate Engagement Plan")

# Initialize models and retrievers once
@st.cache_resource
def load_pipeline():
    llm = OpenAI(temperature=0.3)
    embeddings = OpenAIEmbeddings()
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    # Company retriever
    loader = DirectoryLoader("db/company")
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, embeddings)
    company_retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    company_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=company_retriever,
        input_key="query",
        output_key="company_profile"
    )

    # Research chain
    research_prompt = PromptTemplate(
        input_variables=["company_profile","project_brief"],
        template="""
You are a revenue strategist. Based on the company profile and project brief, list 3–5 revenue streams aligned with the client's needs.

Company Profile:
{company_profile}

Project Brief:
{project_brief}
"""
    )
    research_chain = LLMChain(llm=llm, prompt=research_prompt, output_key="revenue_streams")

    # Stakeholder chain
    stakeholder_prompt = PromptTemplate(
        input_variables=["persona_text","history_text"],
        template="""
You are an expert relationship manager. Given this persona and interaction history, craft a concise stakeholder profile:

Persona:
{persona_text}

History:
{history_text}
"""
    )
    stakeholder_chain = LLMChain(llm=llm, prompt=stakeholder_prompt, output_key="stakeholder_profile")

    # Planning chain
    planning_prompt = PromptTemplate(
        input_variables=["revenue_streams","stakeholder_profile"],
        template="""
You are a strategic planner. Given these revenue streams and stakeholder profile, outline a step-by-step engagement plan:
1. Approach
2. Maintain
3. Position each stream

Revenue Streams:
{revenue_streams}

Stakeholder Profile:
{stakeholder_profile}
"""
    )
    planning_chain = LLMChain(llm=llm, prompt=planning_prompt, output_key="engagement_plan")

    return company_chain, research_chain, stakeholder_chain, planning_chain

company_chain, research_chain, stakeholder_chain, planning_chain = load_pipeline()

# Main execution
if run_button:
    if not client_name or not person_name or not project_brief:
        st.error("Please fill out all inputs in the sidebar.")
    else:
        with st.spinner("Running analysis..."):
            # Company Profile
            company_profile = company_chain.run(query="company overview")

            # Load persona and history
            try:
                persona_path = os.path.join("db/personas", f"persona_{person_name}.md")
                with open(persona_path, 'r', encoding='utf-8') as f:
                    persona_text = f.read()
            except FileNotFoundError:
                persona_text = f"[Error] Persona file not found: persona_{person_name}.md"

            try:
                history_path = os.path.join("db/histories", f"history_{client_name}.txt")
                with open(history_path, 'r', encoding='utf-8') as f:
                    history_text = f.read()
            except FileNotFoundError:
                history_text = f"[Error] History file not found: history_{client_name}.txt"

            # Research
            revenue_streams = research_chain.run(
                company_profile=company_profile,
                project_brief=project_brief
            )

            # Stakeholder
            stakeholder_profile = stakeholder_chain.run(
                persona_text=persona_text,
                history_text=history_text
            )

            # Planning
            engagement_plan = planning_chain.run(
                revenue_streams=revenue_streams,
                stakeholder_profile=stakeholder_profile
            )

        # Display results
        st.header("Company Profile")
        st.write(company_profile)
        st.header("Persona")
        st.write(persona_text)
        st.header("History")
        st.write(history_text)
        st.header("Potential Revenue Streams")
        st.write(revenue_streams)
        st.header("Stakeholder Profile")
        st.write(stakeholder_profile)
        st.header("Engagement Plan")
        st.write(engagement_plan)
