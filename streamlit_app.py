import streamlit as st
import os
from typing import Dict, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="PubMed Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .search-box {
        margin: 1rem 0;
    }
    .paper-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #2E86AB;
    }
    .paper-title {
        color: #1f4e79;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .paper-meta {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .abstract-text {
        margin-top: 0.5rem;
        text-align: justify;
        line-height: 1.5;
    }
    .example-queries {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize Groq LLM
@st.cache_resource
def init_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


# PubMed API Tools (same as original)
@tool
def search_papers_by_author(author_name: str) -> str:
    """Search for papers by a specific author using PubMed API."""
    cleaned_name = author_name.strip()
    titles = ["Dr.", "Dr", "Prof.", "Prof", "Professor", "Mr.", "Ms.", "Mrs."]
    for title in titles:
        if cleaned_name.startswith(title + " "):
            cleaned_name = cleaned_name[len(title) :].strip()

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": f"{cleaned_name}[Author]",
        "retmax": 10,
        "retmode": "json",
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        id_list = data.get("esearchresult", {}).get("idlist", [])
        count = data.get("esearchresult", {}).get("count", "0")

        if count == "0":
            last_name = (
                cleaned_name.split()[-1] if cleaned_name.split() else cleaned_name
            )
            params["term"] = f"{last_name}[Author]"
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            count = data.get("esearchresult", {}).get("count", "0")

        return f"Found {count} papers by {cleaned_name}. Paper IDs: {', '.join(id_list[:10])}"
    except Exception as e:
        return f"Error searching papers: {str(e)}"


@tool
def search_paper_by_title(title: str) -> str:
    """Search for papers by title using PubMed API."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": f'"{title}"[Title]',
        "retmax": 5,
        "retmode": "json",
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        id_list = data.get("esearchresult", {}).get("idlist", [])

        if id_list:
            return get_paper_details.invoke({"paper_id": id_list[0]})
        else:
            params["term"] = f"{title}[Title]"
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])

            if id_list:
                return get_paper_details.invoke({"paper_id": id_list[0]})

        return "No papers found with this title."
    except Exception as e:
        return f"Error searching paper by title: {str(e)}"


@tool
def get_paper_details(paper_id: str) -> str:
    """Get detailed information about a paper by its PubMed ID."""
    if not paper_id.isdigit():
        return f"Invalid paper ID: {paper_id}"

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": paper_id, "retmode": "xml"}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        article = root.find(".//PubmedArticle")

        if not article:
            return f"No article found for ID {paper_id}"

        title = article.findtext(".//ArticleTitle", "No title")

        # Extract abstract
        abstract_texts = article.findall(".//AbstractText")
        if abstract_texts:
            abstract_parts = []
            for text in abstract_texts:
                label = text.get("Label", "")
                content = text.text or ""
                if label:
                    abstract_parts.append(f"{label}: {content}")
                else:
                    abstract_parts.append(content)
            abstract = " ".join(abstract_parts)
        else:
            abstract = "No abstract available"

        # Extract authors
        authors = []
        for author in article.findall(".//Author"):
            last_name = author.findtext("LastName", "")
            fore_name = author.findtext("ForeName", "")
            if last_name or fore_name:
                authors.append(f"{fore_name} {last_name}".strip())

        journal = article.findtext(".//Journal/Title", "Unknown journal")
        year = article.findtext(".//PubDate/Year", "Unknown year")

        # Extract DOI
        doi = None
        for id_elem in article.findall(".//ArticleId"):
            if id_elem.get("IdType") == "doi":
                doi = id_elem.text
                break

        return {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "doi": doi,
            "abstract": abstract,
        }

    except Exception as e:
        return f"Error fetching paper details: {str(e)}"


# Research Agent Class (simplified for Streamlit)
class StreamlitResearchAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = {
            "search_papers_by_author": search_papers_by_author,
            "get_paper_details": get_paper_details,
            "search_paper_by_title": search_paper_by_title,
        }

    def parse_tool_call(self, llm_output: str) -> tuple:
        tool_pattern = r'Tool:\s*(\w+)\("([^"]+)"\)'
        match = re.search(tool_pattern, llm_output)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def execute_tool(self, tool_name: str, argument: str):
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            if "author" in tool_name:
                return tool.invoke({"author_name": argument})
            elif tool_name == "get_paper_details":
                return tool.invoke({"paper_id": argument})
            elif tool_name == "search_paper_by_title":
                return tool.invoke({"title": argument})
        return f"Unknown tool: {tool_name}"

    def process_query(self, user_input: str):
        # Check for direct paper ID
        id_match = re.search(r"\b(\d{7,9})\b", user_input)
        if id_match and any(
            word in user_input.lower()
            for word in ["details", "paper id", "tell me about", "get"]
        ):
            paper_id = id_match.group(1)
            return self.execute_tool("get_paper_details", paper_id)

        # LLM decision making
        prompt = f"""You are a research paper assistant. Based on the user's request, decide which tool to use.

Available tools:
- search_papers_by_author("author name") - Search for papers by an author
- get_paper_details("paper_id") - Get details about a specific paper
- search_paper_by_title("paper title") - Search for a paper by its title

User request: {user_input}

Respond with EXACTLY ONE of these formats:
1. Tool: search_papers_by_author("Author Name")
2. Tool: get_paper_details("12345678")
3. Tool: search_paper_by_title("Paper Title")
4. Direct: [your answer]

Your response:"""

        response = self.llm.invoke(prompt)
        llm_output = response.content

        if llm_output.startswith("Direct:"):
            return llm_output[7:].strip()

        tool_name, argument = self.parse_tool_call(llm_output)
        if tool_name:
            return self.execute_tool(tool_name, argument)

        return "I couldn't understand your request. Please try again."


# Streamlit UI
def main():
    # Header
    st.markdown(
        '<h1 class="main-header">📚 PubMed Research Assistant</h1>',
        unsafe_allow_html=True,
    )

    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ Please set your GROQ_API_KEY environment variable!")
        st.stop()

    # Initialize agent
    llm = init_llm()
    agent = StreamlitResearchAgent(llm)

    # Example queries
    with st.expander("💡 Example Queries", expanded=False):
        st.markdown(
            """
        <div class="example-queries">
        <strong>Try these example searches:</strong><br>
        • "Show papers by Dr. Debasisa Mohanty"<br>
        • "Research papers by Dr. Gitanjali Yadav"<br>
        • "Get details for paper ID 37635766"<br>
        • "Tell me about paper 40125545"<br>
        • "Find paper about HgutMgene-Miner"
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Search interface
    st.markdown('<div class="search-box">', unsafe_allow_html=True)

    # Input methods
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Enter your research query:",
            placeholder="Search by author, title, or paper ID...",
            label_visibility="collapsed",
        )

    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Process query
    if search_button and query:
        with st.spinner("Searching PubMed database..."):
            try:
                result = agent.process_query(query)
                display_results(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Quick action buttons
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔬 Random Paper", use_container_width=True):
            with st.spinner("Fetching a random paper..."):
                # Get a random recent paper (simplified)
                result = agent.execute_tool("get_paper_details", "40125545")
                display_results(result)

    with col2:
        if st.button("📊 Popular Authors", use_container_width=True):
            st.info("Try searching for: Nature, Science, Cell, NEJM authors")

    with col3:
        if st.button("ℹ️ Help", use_container_width=True):
            st.info(
                """
            **How to use:**
            - Search by author: "papers by Dr. Smith"
            - Search by title: "machine learning in medicine"
            - Get paper details: "paper ID 12345678"
            """
            )


def display_results(result):
    """Display search results in a formatted way."""
    if isinstance(result, dict):
        # Single paper result
        display_paper_card(result)
    elif isinstance(result, str):
        if "Found" in result and "papers by" in result:
            # Search result with IDs
            st.success(result)

            # Extract paper IDs and show first few papers
            paper_ids = re.findall(r"\b(\d{7,9})\b", result)
            if paper_ids:
                st.markdown("### 📄 Paper Details")
                for i, paper_id in enumerate(paper_ids[:3]):  # Show first 3 papers
                    with st.spinner(f"Loading paper {i+1}..."):
                        paper_details = get_paper_details.invoke({"paper_id": paper_id})
                        if isinstance(paper_details, dict):
                            display_paper_card(paper_details)
        else:
            # Other string results
            if "Error" in result:
                st.error(result)
            else:
                st.info(result)
    else:
        st.write(result)


def display_paper_card(paper_data):
    """Display a single paper in a card format."""
    if not isinstance(paper_data, dict):
        st.write(paper_data)
        return

    with st.container():
        st.markdown(
            f"""
        <div class="paper-card">
            <div class="paper-title">{paper_data.get('title', 'No title')}</div>
            <div class="paper-meta">
                <strong>Authors:</strong> {', '.join(paper_data.get('authors', [])) if paper_data.get('authors') else 'No authors listed'}<br>
                <strong>Journal:</strong> {paper_data.get('journal', 'Unknown')} ({paper_data.get('year', 'Unknown year')})<br>
                <strong>Paper ID:</strong> {paper_data.get('paper_id', 'N/A')}<br>
                <strong>DOI:</strong> {paper_data.get('doi', 'Not available')}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Abstract in expandable section
        if (
            paper_data.get("abstract")
            and paper_data["abstract"] != "No abstract available"
        ):
            with st.expander("📝 Abstract", expanded=False):
                st.markdown(
                    f'<div class="abstract-text">{paper_data["abstract"]}</div>',
                    unsafe_allow_html=True,
                )

        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if paper_data.get("doi"):
                st.markdown(f"[🔗 View Paper](https://doi.org/{paper_data['doi']})")
        with col2:
            if paper_data.get("paper_id"):
                st.markdown(
                    f"[📚 PubMed](https://pubmed.ncbi.nlm.nih.gov/{paper_data['paper_id']}/)"
                )
        with col3:
            if st.button(
                f"📋 Copy ID", key=f"copy_{paper_data.get('paper_id', 'unknown')}"
            ):
                st.success(f"Paper ID {paper_data.get('paper_id')} copied!")


if __name__ == "__main__":
    main()
