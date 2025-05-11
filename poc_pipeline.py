'''
Proof-of-Concept: Multi-Agent Pipeline for Client Engagement with Direct Persona/History Loading

Dependencies:
  pip install langchain-openai faiss-cpu python-dotenv

Setup:
  1. Create a `.env` file at your project root with:
        OPENAI_API_KEY=your_api_key_here

  2. Create these directories:
        mkdir -p db/company
        mkdir -p db/personas
        mkdir -p db/histories

  3. Populate each folder with your docs, using these naming conventions:
     - **db/company/**: any `.txt`/`.md`, e.g. `company_profile.txt`
     - **db/personas/**: `persona_{person_name}.md` (e.g. `persona_alice.md`)
     - **db/histories/**: `history_{client_name}.txt` (e.g. `history_acmecorp.txt`)

Usage:
  python poc_pipeline.py

  You will be prompted for:
    1. Client Name  (must match `history_{client_name}.txt`)
    2. Person Name  (must match `persona_{person_name}.md`)
    3. Project brief / needs
'''

import os
from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

# Load API key
env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(env_path)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")

# Initialize LLM & embeddings
llm = OpenAI(temperature=0.3)
embeddings = OpenAIEmbeddings()
splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# Build a FAISS retriever for company docs
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

# Prompt templates for later agents
research_prompt = PromptTemplate(
    input_variables=["company_profile", "project_brief"],
    template="""
You are a revenue strategist. Based on the company profile and project brief, list 3–5 revenue streams aligned with the client's needs.

Company Profile:
{company_profile}

Project Brief:
{project_brief}
"""
)
research_chain = LLMChain(llm=llm, prompt=research_prompt, output_key="revenue_streams")

stakeholder_prompt = PromptTemplate(
    input_variables=["persona_text", "history_text"],
    template="""
You are an expert relationship manager. Given this persona and interaction history, craft a concise stakeholder profile:

Persona:
{persona_text}

History:
{history_text}
"""
)
stakeholder_chain = LLMChain(llm=llm, prompt=stakeholder_prompt, output_key="stakeholder_profile")

planning_prompt = PromptTemplate(
    input_variables=["revenue_streams", "stakeholder_profile"],
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

# Orchestrator
if __name__ == "__main__":
    print("--- Client Engagement Pipeline POC ---")
    client_name   = input("1. Enter Client Name (history_{client_name}.txt): ")
    person_name   = input("2. Enter Person Name (persona_{person_name}.md): ")
    project_brief = input("3. Enter project brief / needs: ")

    # 1. Company Profile Retrieval
    company_profile = company_chain.run(query="company overview")

    # 2. Direct file load for persona & history
    persona_path = os.path.join("db", "personas", f"persona_{person_name}.md")
    history_path = os.path.join("db", "histories", f"history_{client_name}.txt")
    try:
        with open(persona_path, 'r', encoding='utf-8') as f:
            persona_text = f.read()
    except FileNotFoundError:
        persona_text = f"[Error] Persona file not found: {persona_path}"

    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history_text = f.read()
    except FileNotFoundError:
        history_text = f"[Error] History file not found: {history_path}"

    # 3. Research
    revenue_streams = research_chain.run(
        company_profile=company_profile,
        project_brief=project_brief
    )

    # 4. Stakeholder Profile
    stakeholder_profile = stakeholder_chain.run(
        persona_text=persona_text,
        history_text=history_text
    )

    # 5. Engagement Plan
    engagement_plan = planning_chain.run(
        revenue_streams=revenue_streams,
        stakeholder_profile=stakeholder_profile
    )

    # Display results
    print(f"\n[Company Profile]\n{company_profile}\n")
    print(f"[Persona]\n{persona_text}\n")
    print(f"[History]\n{history_text}\n")
    print(f"[Revenue Streams]\n{revenue_streams}\n")
    print(f"[Stakeholder Profile]\n{stakeholder_profile}\n")
    print(f"[Engagement Plan]\n{engagement_plan}\n")
