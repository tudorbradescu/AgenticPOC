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
env_path = os.path.join(os.getcwd(), ".env")
load_dotenv(env_path)
if not os.getenv("OPENAI_API_KEY"):
    st.error("Please set OPENAI_API_KEY in your .env file.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.title("Client Engagement Dashboard")
flow_type = st.sidebar.selectbox("Select Flow", ["Account Navigation Flow", "Opportunity Flow", "Social Media Flow"])

# Core identifiers
client_name   = st.sidebar.text_input("Client Name", help="Matches history_{client_name}.txt")
person_name   = st.sidebar.text_input("Person Name", help="Matches persona_{person_name}.md")
project_brief = st.sidebar.text_area("Project Brief / Needs")

# Detailed context expanders
with st.sidebar.expander("Client Meetings (chronological)"):
    client_meetings = st.text_area("Enter date + notes", height=120)
with st.sidebar.expander("Team Meetings (internal)"):
    team_meetings = st.text_area("Enter attendee inputs", height=120)
with st.sidebar.expander("QBR Meetings"):
    qbr_meetings = st.text_area("Enter QBR details", height=150)
with st.sidebar.expander("Client Background"):
    client_background = st.text_area("Background & intro context", height=120)
with st.sidebar.expander("Client Persona"):
    client_persona = st.text_area("Persona narrative", height=120)
with st.sidebar.expander("Client History Overview"):
    client_history = st.text_area("Chronological account history", height=120)
with st.sidebar.expander("My Company Strategy"):
    company_strategy = st.text_area("Strategic goals", height=100)
with st.sidebar.expander("Skill Matrix"):
    skill_matrix = st.text_area("Key team skills", height=100)
with st.sidebar.expander("Case Studies"):
    case_studies = st.text_area("Notable past projects", height=120)
with st.sidebar.expander("Team Shape & Roles"):
    team_shape = st.text_area("Team org and roles", height=120)
with st.sidebar.expander("SMEs & Domain Knowledge"):
    smes_domain = st.text_area("Subject-matter experts & domains", height=120)
with st.sidebar.expander("Objectives & Sales Team"):
    objectives = st.text_area("Business objectives & sales team info", height=120)
with st.sidebar.expander("My Network (social links)"):
    my_network = st.text_area("Key connections & insights", height=100)

run_button = st.sidebar.button("Generate Plan")

# --- Load chains once ---
@st.cache_resource
def load_chains():
    llm = OpenAI(temperature=0.3)
    emb = OpenAIEmbeddings()
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    # Ingest company docs
    comp_docs = DirectoryLoader("db/company").load()
    comp_chunks = splitter.split_documents(comp_docs)
    comp_index = FAISS.from_documents(comp_chunks, emb)
    comp_ret   = comp_index.as_retriever(search_kwargs={"k":5})
    company_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                               retriever=comp_ret, input_key="query",
                                               output_key="company_profile")

    # QBR agent: use client_meetings + team_meetings + qbr_meetings
    qbr_prompt = PromptTemplate(
        input_variables=["meetings_notes"],
        template="""
You are a QBR analyst. Given these meeting notes (client, internal, QBR), extract wins, risks, and opportunities.

Notes:
{meetings_notes}
"""
    )
    qbr_chain = LLMChain(llm=llm, prompt=qbr_prompt, output_key="qbr_insights")

    # Business mapping: company_profile + qbr_insights + case_studies + skill_matrix + company_strategy
    bm_prompt = PromptTemplate(
        input_variables=["company_profile","qbr_insights","case_studies","skill_matrix","company_strategy"],
        template="""
You are a business mapper. Based on company profile, QBR insights, case studies, skill matrix, and company strategy, map:
- Leverage points
- Market positioning
- Strengths and weaknesses

Company Profile:
{company_profile}

QBR Insights:
{qbr_insights}

Case Studies:
{case_studies}

Skill Matrix:
{skill_matrix}

Company Strategy:
{company_strategy}
"""
    )
    bm_chain = LLMChain(llm=llm, prompt=bm_prompt, output_key="bm_map")

    # Introductions: uses bm_map + team_shape + client_persona
    intro_prompt = PromptTemplate(
        input_variables=["bm_map","team_shape","client_persona"],
        template="""
You are an introductions specialist. Using business mapping, team shape, and persona, craft a targeted stakeholder intro strategy.

Business Mapping:
{bm_map}

Team Shape & Roles:
{team_shape}

Client Persona:
{client_persona}
"""
    )
    intro_chain = LLMChain(llm=llm, prompt=intro_prompt, output_key="intro_strategy")

    # Research (Opportunity): company_profile + project_brief + client_background
    research_prompt = PromptTemplate(
        input_variables=["company_profile","project_brief","client_background"],
        template="""
You are a revenue strategist. Given company profile, client background, and project brief, list 3–5 revenue streams.

Company Profile:
{company_profile}

Client Background:
{client_background}

Project Brief:
{project_brief}
"""
    )
    research_chain = LLMChain(llm=llm, prompt=research_prompt, output_key="revenue_streams")

    # Stakeholder: client_persona + client_history + objectives + smes_domain
    stkh_prompt = PromptTemplate(
        input_variables=["client_persona","client_history","objectives","smes_domain"],
        template="""
You are a stakeholder profiler. Based on persona, historical overview, objectives, and domain expertise, summarize the stakeholder profile.

Persona:
{client_persona}

History:
{client_history}

Objectives:
{objectives}

Domain Knowledge:
{smes_domain}
"""
    )
    stkh_chain = LLMChain(llm=llm, prompt=stkh_prompt, output_key="stakeholder_profile")

    # Client Meeting: revenue_streams + stakeholder_profile + team_meetings + objectives
    cm_prompt = PromptTemplate(
        input_variables=["revenue_streams","stakeholder_profile","team_meetings","objectives"],
        template="""
You are a client meeting planner. Using revenue streams, stakeholder profile, team inputs, and objectives, outline agenda and next steps.

Revenue Streams:
{revenue_streams}

Stakeholder Profile:
{stakeholder_profile}

Team Inputs:
{team_meetings}

Objectives:
{objectives}
"""
    )
    cm_chain = LLMChain(llm=llm, prompt=cm_prompt, output_key="meeting_plan")

    # Social Media: smes_domain + my_network + comp_profile
    social_prompt = PromptTemplate(
        input_variables=["smes_domain","my_network","company_profile"],
        template="""
You are a social media strategist. Based on domain strengths, network connections, and company profile, suggest social outreach tactics.

Domain Strengths:
{smes_domain}

Network Insights:
{my_network}

Company Profile:
{company_profile}
"""
    )
    social_chain = LLMChain(llm=llm, prompt=social_prompt, output_key="social_plan")

    return dict(
        company_chain=company_chain,
        qbr_chain=qbr_chain,
        bm_chain=bm_chain,
        intro_chain=intro_chain,
        research_chain=research_chain,
        stkh_chain=stkh_chain,
        cm_chain=cm_chain,
        social_chain=social_chain
    )

chains = load_chains()

# --- Main Execution ---
if run_button:
    if not client_name or not person_name:
        st.error("Please enter Client Name & Person Name.")
    else:
        with st.spinner("Processing..."):
            # Company profile retrieval
            cp = chains['company_chain'].run(query="company overview")
            # Aggregate QBR/meeting notes
            meetings = "\n".join(filter(None, [client_meetings, team_meetings, qbr_meetings]))
            qbr_out = chains['qbr_chain'].run(meetings_notes=meetings)
            bm_out  = chains['bm_chain'].run(company_profile=cp,
                                            qbr_insights=qbr_out,
                                            case_studies=case_studies,
                                            skill_matrix=skill_matrix,
                                            company_strategy=company_strategy)
            intro_out = chains['intro_chain'].run(bm_map=bm_out,
                                                 team_shape=team_shape,
                                                 client_persona=client_persona)

            # Opportunity flow
            rs_out = chains['research_chain'].run(company_profile=cp,
                                                 project_brief=project_brief,
                                                 client_background=client_background)
            stkh_out = chains['stkh_chain'].run(client_persona=client_persona,
                                                client_history=client_history,
                                                objectives=objectives,
                                                smes_domain=smes_domain)
            cm_out   = chains['cm_chain'].run(revenue_streams=rs_out,
                                              stakeholder_profile=stkh_out,
                                              team_meetings=team_meetings,
                                              objectives=objectives)

            # Social media flow
            social_out = chains['social_chain'].run(smes_domain=smes_domain,
                                                     my_network=my_network,
                                                     company_profile=cp)
        
        # Display
        st.header("Company Profile")
        st.write(cp)
        st.header("QBR Insights")
        st.write(qbr_out)
        st.header("Business Mapping")
        st.write(bm_out)
        st.header("Introductions Strategy")
        st.write(intro_out)
        st.header("Revenue Streams")
        st.write(rs_out)
        st.header("Stakeholder Profile")
        st.write(stkh_out)
        st.header("Client Meeting Plan")
        st.write(cm_out)
        st.header("Social Media Outreach Plan")
        st.write(social_out)
