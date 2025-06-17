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

# Initialize Groq LLM
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


# PubMed API Tools
@tool
def search_papers_by_author(author_name: str) -> str:
    """Search for papers by a specific author using PubMed API."""
    # Clean up the author name - remove titles like Dr., Prof., etc.
    cleaned_name = author_name.strip()
    titles = ["Dr.", "Dr", "Prof.", "Prof", "Professor", "Mr.", "Ms.", "Mrs."]
    for title in titles:
        if cleaned_name.startswith(title + " "):
            cleaned_name = cleaned_name[len(title) :].strip()

    print(
        f"[Tool] Searching papers for author: {cleaned_name} (original: {author_name})"
    )
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": f"{cleaned_name}[Author]",
        "retmax": 20,
        "retmode": "json",
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        id_list = data.get("esearchresult", {}).get("idlist", [])
        count = data.get("esearchresult", {}).get("count", "0")

        if count == "0":
            # Try with just last name
            last_name = (
                cleaned_name.split()[-1] if cleaned_name.split() else cleaned_name
            )
            params["term"] = f"{last_name}[Author]"
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            count = data.get("esearchresult", {}).get("count", "0")

            if count != "0":
                return f"Found {count} papers by authors with last name '{last_name}'. Paper IDs: {', '.join(id_list[:10])}"

        return f"Found {count} papers by {cleaned_name}. Paper IDs: {', '.join(id_list[:10])}"
    except Exception as e:
        return f"Error searching papers: {str(e)}"


@tool
def search_paper_by_title(title: str) -> str:
    """Search for papers by title using PubMed API."""
    print(f"[Tool] Searching for paper with title: {title[:50]}...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": f'"{title}"[Title]',  # Exact title search
        "retmax": 5,
        "retmode": "json",
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        id_list = data.get("esearchresult", {}).get("idlist", [])
        count = data.get("esearchresult", {}).get("count", "0")

        if id_list:
            print(
                f"[Tool] Found {count} paper(s) matching title. Getting details for first match..."
            )
            # Automatically get details for the first result
            return get_paper_details.invoke({"paper_id": id_list[0]})
        else:
            # Try partial title search if exact match fails
            params["term"] = f"{title}[Title]"  # Remove quotes for partial match
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])

            if id_list:
                print(
                    f"[Tool] Found {len(id_list)} paper(s) with partial title match. Getting details for first match..."
                )
                return get_paper_details.invoke({"paper_id": id_list[0]})

        return "No papers found with this title. Try searching by author or use a shorter title."
    except Exception as e:
        return f"Error searching paper by title: {str(e)}"


@tool
def get_paper_details(paper_id: str) -> str:
    """Get detailed information about a paper by its PubMed ID."""
    print(f"[Tool] Getting details for paper ID: {paper_id}")

    # Validate paper ID
    if not paper_id.isdigit():
        return f"Invalid paper ID: {paper_id}. Paper IDs must be numeric."

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": paper_id, "retmode": "xml"}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        article = root.find(".//PubmedArticle")

        if not article:
            return f"No article found for ID {paper_id}"

        # Extract all information
        title = article.findtext(".//ArticleTitle", "No title")

        # Extract full abstract
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

        # Extract ALL authors
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

        result = f"""
PAPER DETAILS:
==============
Paper ID: {paper_id}
Title: {title}
Authors: {', '.join(authors) if authors else 'No authors listed'}
Journal: {journal} ({year})
DOI: {doi or 'Not available'}

ABSTRACT:
{abstract}
"""
        return result

    except Exception as e:
        return f"Error fetching paper details: {str(e)}"


# Simple ReAct Agent Class
class SimpleResearchAgent:
    def __init__(self):
        self.llm = llm
        self.tools = {
            "search_papers_by_author": search_papers_by_author,
            "get_paper_details": get_paper_details,
            "search_paper_by_title": search_paper_by_title,
        }
        self.conversation_history = []
        self.paper_id_cache = {}  # Cache to remember paper IDs

    def parse_tool_call(self, llm_output: str) -> tuple:
        """Parse the LLM output to extract tool name and arguments."""
        # Look for patterns like: Tool: search_papers_by_author("Dr. Name")
        tool_pattern = r'Tool:\s*(\w+)\("([^"]+)"\)'
        match = re.search(tool_pattern, llm_output)

        if match:
            tool_name = match.group(1)
            argument = match.group(2)
            return tool_name, argument
        return None, None

    def execute_tool(self, tool_name: str, argument: str) -> str:
        """Execute the specified tool with the given argument."""
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            if "author" in tool_name:
                return tool.invoke({"author_name": argument})
            elif tool_name == "get_paper_details":
                return tool.invoke({"paper_id": argument})
            elif tool_name == "search_paper_by_title":
                return tool.invoke({"title": argument})  # ← Added this condition
        return f"Unknown tool: {tool_name}"

    def think_and_act(self, user_input: str) -> str:
        """Main ReAct loop: Think, Act, Observe, Respond."""
        # Add user input to history
        self.conversation_history.append(f"User: {user_input}")

        # Check if user is asking about a specific paper by ID
        id_match = re.search(r"\b(\d{7,9})\b", user_input)
        if id_match and any(
            word in user_input.lower()
            for word in ["details", "paper id", "tell me about", "get"]
        ):
            paper_id = id_match.group(1)
            print(f"[Agent] Detected paper ID request: {paper_id}")
            result = get_paper_details.invoke({"paper_id": paper_id})
            self.conversation_history.append(f"Tool Result: {result}")
            return self.format_response(result)

        # Check if asking about a paper title we've seen before
        if "cavity architecture" in user_input.lower() and "37635766" in str(
            self.paper_id_cache
        ):
            print("[Agent] Using cached paper ID for Cavity architecture paper")
            result = get_paper_details.invoke({"paper_id": "37635766"})
            self.conversation_history.append(f"Tool Result: {result}")
            return self.format_response(result)

        # Create prompt for the LLM
        prompt = f"""You are a research paper assistant. Based on the user's request, decide which tool to use.

        Available tools:
        - search_papers_by_author("author name") - Search for papers by an author
        - get_paper_details("paper_id") - Get details about a specific paper (ID must be numeric)
        - search_paper_by_title("paper title") - Search for a paper by its title  # ← Added this line

        User request: {user_input}

        Recent conversation:
        {chr(10).join(self.conversation_history[-3:])}

        Respond with EXACTLY ONE of these formats:
        1. If searching by author: Tool: search_papers_by_author("Author Name")
        2. If getting paper details by ID: Tool: get_paper_details("12345678")
        3. If searching by paper title: Tool: search_paper_by_title("Paper Title")  # ← Added this line
        4. If you can answer without tools: Direct: [your answer]  # ← Changed from 3 to 4

        Your response:"""

        # Get LLM decision
        response = self.llm.invoke(prompt)
        llm_output = response.content
        print(f"[Agent] LLM decision: {llm_output[:100]}...")

        # Check if direct response
        if llm_output.startswith("Direct:"):
            return llm_output[7:].strip()

        # Parse and execute tool
        tool_name, argument = self.parse_tool_call(llm_output)

        if tool_name:
            print(f"[Agent] Executing: {tool_name}({argument})")
            result = self.execute_tool(tool_name, argument)
            self.conversation_history.append(
                f"Tool: {tool_name}, Result: {result[:200]}..."
            )

            # Cache paper IDs from search results
            if "Paper IDs:" in result:
                ids = re.findall(r"\b(\d{7,9})\b", result)
                for i, pid in enumerate(ids[:5]):
                    self.paper_id_cache[f"paper_{i+1}"] = pid

            # If this was a search, automatically get details for first 2-3 papers
            if tool_name == "search_papers_by_author" and "Paper IDs:" in result:
                paper_ids = re.findall(r"\b(\d{7,9})\b", result)[:3]
                detailed_results = [result]

                for pid in paper_ids:
                    details = get_paper_details.invoke({"paper_id": pid})
                    detailed_results.append(details)

                return self.format_multiple_papers(
                    "\n\n".join(detailed_results), len(paper_ids)
                )

            return self.format_response(result)

        return (
            "I couldn't understand your request. Please try again with a clearer query."
        )

    def format_response(self, tool_result: str) -> str:
        """Format the tool result into a nice response."""
        if "PAPER DETAILS:" in tool_result:
            return tool_result  # Already well formatted
        elif "Found" in tool_result and "papers by" in tool_result:
            return f"📚 {tool_result}\n\nWould you like me to show details for any of these papers?"
        else:
            return tool_result

    def format_multiple_papers(self, combined_results: str, count: int) -> str:
        """Format multiple paper results nicely."""
        lines = combined_results.split("\n")
        summary_line = next((line for line in lines if "Found" in line), "")

        response = f"📚 {summary_line}\n\n"
        response += f"Here are details for the first {count} papers:\n"
        response += "=" * 50 + "\n"

        # Extract and format each paper
        papers = combined_results.split("PAPER DETAILS:")
        for i, paper in enumerate(
            papers[1:], 1
        ):  # Skip the first split (search result)
            response += f"\n📄 PAPER {i}:\n"
            response += paper.strip()
            if i < len(papers) - 1:
                response += "\n" + "-" * 50 + "\n"

        response += "\n\n💡 Need more papers or specific details? Just ask!"
        return response


# Main execution
if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: Please set your GROQ_API_KEY environment variable!")
        exit(1)

    print("PubMed Research Assistant")
    print("=" * 40)
    print("Examples:")
    print("- 'Show papers by Dr. Debasisa Mohanty'")
    print("- 'Research paper of by Dr. Gitanjali Yadav'")
    print("- 'Get details for paper ID 37635766'")
    print("- 'Tell me about paper 40125545'")
    print(
        "- 'Tell me about this paper 'HgutMgene-Miner: In silico genome mining tool for deciphering the drug-metabolizing potential of human gut microbiome''"
    )
    print("\nType 'quit' to exit.\n")

    agent = SimpleResearchAgent()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye! 👋")
            break

        if user_input:
            print("\nAgent: ", end="", flush=True)
            response = agent.think_and_act(user_input)
            print(response)
            print()
