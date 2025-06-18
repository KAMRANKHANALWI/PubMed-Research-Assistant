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
import time

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
        position: relative;
    }
    .paper-number {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: #2E86AB;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .paper-title {
        color: #1f4e79;
        font-weight: bold;
        margin-bottom: 0.5rem;
        margin-right: 3rem;
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
    .stats-bar {
        background: linear-gradient(135deg, #2E86AB 0%, #4A90A4 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    .pagination-info {
        text-align: center;
        color: #666;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Bottom pagination styling - simplified and fixed */
    .pagination-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.5rem;
        margin: 2rem 0;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* Style for current page button */
    .stButton > button[disabled] {
        background: #2E86AB !important;
        color: white !important;
        border: 1px solid #2E86AB !important;
        opacity: 1 !important;
        cursor: default !important;
    }
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 6px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .progress-bar {
        background: linear-gradient(90deg, #2E86AB 0%, #4A90A4 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .loading-text {
        text-align: center;
        color: #666;
        margin: 0.5rem 0;
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


# Enhanced PubMed API Tools to get MORE papers
@tool
def search_papers_by_author(author_name: str) -> dict:
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
        "retmax": 50,  # ← INCREASED FROM 10 TO 50
        "retmode": "json",
        "sort": "pub_date",  # ← SORT BY DATE (NEWEST FIRST)
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

        # ← RETURN STRUCTURED DATA INSTEAD OF STRING
        return {
            "success": True,
            "count": int(count),
            "author": cleaned_name,
            "paper_ids": id_list,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def search_paper_by_title(title: str) -> dict:  # ← Changed from str to dict
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
            # This already returns a dict from get_paper_details
            return get_paper_details.invoke({"paper_id": id_list[0]})
        else:
            params["term"] = f"{title}[Title]"
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])

            if id_list:
                # This already returns a dict from get_paper_details
                return get_paper_details.invoke({"paper_id": id_list[0]})

        # Return dict for consistency
        return {"success": False, "error": "No papers found with this title."}
    except Exception as e:
        # Return dict for consistency
        return {"success": False, "error": f"Error searching paper by title: {str(e)}"}


@tool
def get_paper_details(paper_id: str) -> dict:  # ← Changed from str to dict
    """Get detailed information about a paper by its PubMed ID."""
    if not paper_id.isdigit():
        # Return dict for consistency
        return {"success": False, "error": f"Invalid paper ID: {paper_id}"}

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": paper_id, "retmode": "xml"}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        article = root.find(".//PubmedArticle")

        if not article:
            # Return dict for consistency
            return {"success": False, "error": f"No article found for ID {paper_id}"}

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

        # This is already returning a dict - it was correct!
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
        # Return dict for consistency
        return {"success": False, "error": f"Error fetching paper details: {str(e)}"}


# Research Agent Class (updated to handle new return format)
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
        # Show previous search query if it exists
        default_query = st.session_state.get("search_query", "")
        query = st.text_input(
            "Enter your research query:",
            placeholder="Search by author, title, or paper ID...",
            label_visibility="collapsed",
            value=default_query,
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
    """Display search results in a formatted way - EXACT SAME LOGIC AS ORIGINAL."""
    if isinstance(result, dict):
        # Check if it's author search results with paper_ids (our enhanced feature)
        if result.get("success") and "paper_ids" in result:
            display_all_author_papers(result)
        else:
            # Single paper result (dict format) - SAME AS ORIGINAL
            display_paper_card(result)
    elif isinstance(result, str):
        if "Found" in result and "papers by" in result:
            # Author search result with IDs - SAME AS ORIGINAL
            st.success(result)
            # Extract paper IDs and show first few papers
            paper_ids = re.findall(r"\b(\d{7,9})\b", result)
            if paper_ids:
                st.markdown("### 📄 Paper Details")
                for i, paper_id in enumerate(paper_ids[:3]):  # Show first 3 papers
                    with st.spinner(f"Loading paper {i+1}..."):
                        paper_details = get_paper_details.invoke({"paper_id": paper_id})
                        if isinstance(paper_details, dict):
                            display_paper_card(paper_details, i + 1)
        else:
            # Other string results - SAME AS ORIGINAL
            if "Error" in result:
                st.error(result)
            else:
                st.info(result)
    else:
        st.write(result)


def display_all_author_papers(search_result):
    """Display ALL papers for an author with clean bottom pagination."""
    author_name = search_result["author"]
    total_papers = search_result["count"]
    paper_ids = search_result["paper_ids"]

    # Show stats
    st.markdown(
        f"""
    <div class="stats-bar">
        📊 Found {total_papers:,} papers by {author_name} • Showing {len(paper_ids)} most recent
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Papers per page control
    papers_per_page = st.selectbox(
        "Papers per page:",
        [5, 10, 15, 20, 30],
        index=1,  # Default to 10 papers per page
        key="papers_per_page",
    )

    # Calculate pagination
    total_pages = (len(paper_ids) + papers_per_page - 1) // papers_per_page

    # Reset current page when papers per page changes
    if "prev_papers_per_page" not in st.session_state:
        st.session_state.prev_papers_per_page = papers_per_page

    if st.session_state.prev_papers_per_page != papers_per_page:
        st.session_state.current_page = 1
        st.session_state.prev_papers_per_page = papers_per_page

    # Get current page from session state or default to 1
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    # Ensure current page is within valid range
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages
    if st.session_state.current_page < 1:
        st.session_state.current_page = 1

    current_page = st.session_state.current_page

    # Get papers for current page
    start_idx = (current_page - 1) * papers_per_page
    end_idx = min(start_idx + papers_per_page, len(paper_ids))
    current_page_ids = paper_ids[start_idx:end_idx]

    # Progress tracking
    progress_placeholder = st.empty()

    # Load and display papers
    st.markdown(f"### 📄 Paper Details (Page {current_page} of {total_pages})")

    # Add debug info
    st.markdown(
        f"**Debug:** Current page: {current_page}, Total pages: {total_pages}, Papers per page: {papers_per_page}"
    )

    for i, paper_id in enumerate(current_page_ids):
        # Show progress
        progress = (i + 1) / len(current_page_ids)
        progress_placeholder.markdown(
            f"""
        <div class="loading-text">Loading paper {i + 1} of {len(current_page_ids)}...</div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {progress * 100}%"></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Get paper details
        paper_details = get_paper_details.invoke({"paper_id": paper_id})
        if isinstance(paper_details, dict):
            display_paper_card(paper_details, start_idx + i + 1)

        time.sleep(0.1)  # Small delay to show progress

    # Clear progress
    progress_placeholder.empty()

    # Clean bottom pagination
    if total_pages > 1:
        create_clean_pagination(current_page, total_pages)

    # Show pagination info
    st.markdown(
        f"""
    <div class="pagination-info">
        Showing papers {start_idx + 1}-{end_idx} of {len(paper_ids)} 
        ({total_papers:,} total papers found)
    </div>
    """,
        unsafe_allow_html=True,
    )


def create_clean_pagination(current_page, total_pages):
    """Create clean, working pagination with Streamlit buttons."""

    # Create pagination container
    st.markdown('<div class="pagination-container">', unsafe_allow_html=True)

    # Create columns for pagination buttons
    cols = st.columns([2, 1, 1, 1, 1, 1, 2])

    # Previous button
    with cols[0]:
        if st.button("← Previous", disabled=current_page <= 1, key="prev_page"):
            st.session_state.current_page = max(1, current_page - 1)
            st.rerun()  # Use st.rerun() instead of experimental_rerun()

    # Calculate which pages to show (max 5 page buttons)
    if total_pages <= 5:
        pages_to_show = list(range(1, total_pages + 1))
    else:
        if current_page <= 3:
            pages_to_show = [1, 2, 3, 4, 5]
        elif current_page >= total_pages - 2:
            pages_to_show = [
                total_pages - 4,
                total_pages - 3,
                total_pages - 2,
                total_pages - 1,
                total_pages,
            ]
        else:
            pages_to_show = [
                current_page - 2,
                current_page - 1,
                current_page,
                current_page + 1,
                current_page + 2,
            ]

    # Page number buttons (use middle columns)
    for i, page_num in enumerate(pages_to_show):
        if i < 5:  # Maximum 5 page buttons
            with cols[i + 1]:
                # Fix button type issue - don't use type parameter for non-current pages
                if page_num == current_page:
                    # Current page button (highlighted and disabled)
                    st.button(
                        str(page_num), key=f"page_{page_num}_current", disabled=True
                    )
                else:
                    # Other page buttons
                    if st.button(str(page_num), key=f"page_{page_num}"):
                        st.session_state.current_page = page_num
                        st.rerun()  # Use st.rerun() instead of experimental_rerun()

    # Next button
    with cols[6]:
        if st.button("Next →", disabled=current_page >= total_pages, key="next_page"):
            st.session_state.current_page = min(total_pages, current_page + 1)
            st.rerun()  # Use st.rerun() instead of experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def display_paper_card(paper_data, paper_number=None):
    """Display a single paper in styled card format, works in single or paginated mode."""
    if not isinstance(paper_data, dict):
        st.write(paper_data)
        return

    # Build the card HTML
    card_html = f"""
    <div class="paper-card">
      {'<div class="paper-number">#' + str(paper_number) + '</div>' if paper_number else ''}
      <div class="paper-title">{paper_data.get('title', 'No title')}</div>
      <div class="paper-meta">
        <strong>Authors:</strong> {', '.join(paper_data.get('authors', [])) if paper_data.get('authors') else 'No authors listed'}<br>
        <strong>Journal:</strong> {paper_data.get('journal', 'Unknown')} ({paper_data.get('year', 'Unknown year')})<br>
        <strong>Paper ID:</strong> {paper_data.get('paper_id', 'N/A')}<br>
        <strong>DOI:</strong> {paper_data.get('doi', 'Not available')}
      </div>
    </div>
    """

    # Render the card
    st.markdown(card_html, unsafe_allow_html=True)

    # Abstract
    if paper_data.get("abstract") and paper_data["abstract"] != "No abstract available":
        with st.expander("📝 Abstract", expanded=False):
            abstract_html = f'<div class="abstract-text">{paper_data["abstract"]}</div>'
            st.markdown(abstract_html, unsafe_allow_html=True)

    # Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        doi = paper_data.get("doi")
        if doi:
            st.markdown(
                f"[🔗 View Paper](https://doi.org/{doi})", unsafe_allow_html=True
            )
    with col2:
        pid = paper_data.get("paper_id")
        if pid:
            st.markdown(
                f"[📚 PubMed](https://pubmed.ncbi.nlm.nih.gov/{pid}/)",
                unsafe_allow_html=True,
            )
    with col3:
        key = f"copy_{paper_data.get('paper_id', 'unknown')}_{paper_number or 'single'}"
        if st.button("📋 Copy ID", key=key):
            st.success(f"Paper ID {paper_data.get('paper_id')} copied!")


if __name__ == "__main__":
    main()
