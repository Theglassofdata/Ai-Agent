import os
import time
import argparse
import textwrap
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import requests
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_community.tools.tavily_search import TavilySearchResults
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

class SearchAssistant:
    """An enhanced search assistant that combines web search with LLM summarization."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", max_retries: int = 3):
        """Initialize the search assistant with the specified model."""
        self.console = Console()
        
        # Load environment variables
        load_dotenv()
        self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        
        if not self.hf_token:
            self.console.print("[bold red]Error:[/] HUGGINGFACEHUB_API_TOKEN not found in environment variables or .env file")
            exit(1)
            
        if not self.tavily_key:
            self.console.print("[bold red]Error:[/] TAVILY_API_KEY not found in environment variables or .env file")
            exit(1)
        
        # Initialize components
        self.model_name = model_name
        self.max_retries = max_retries
        self._initialize_services()
        
    def _initialize_services(self):
        """Initialize the search and LLM services with error handling."""
        try:
            self.search = TavilySearchResults(
                k=5,  # Get more results for better coverage
                api_key=self.tavily_key
            )
            self.console.print(f"[green]✓[/] Tavily Search initialized")
        except Exception as e:
            self.console.print(f"[bold red]Error initializing Tavily Search:[/] {str(e)}")
            exit(1)
            
        try:
            self.llm_client = InferenceClient(
                model=self.model_name,
                token=self.hf_token
            )
            self.console.print(f"[green]✓[/] LLM client initialized with model: {self.model_name}")
        except Exception as e:
            self.console.print(f"[bold red]Error initializing LLM client:[/] {str(e)}")
            exit(1)
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web with retries and error handling."""
        self.console.print(f"\n[bold blue]🔍 Searching for:[/] {query}")
        
        for attempt in range(self.max_retries):
            try:
                results = self.search.run(query)
                # Debug print to examine result structure
                self.console.print(f"[dim]Debug: Got {len(results)} results[/dim]")
                
                # Process results to ensure they have all required fields
                processed_results = []
                for r in results:
                    # Make sure we have a snippet/content field
                    if not r.get("snippet"):
                        # Try to extract content from different possible fields
                        content = r.get("content") or r.get("body") or r.get("text") or "No content available."
                        r["snippet"] = content[:500] + "..." if len(content) > 500 else content
                    
                    processed_results.append(r)
                
                if processed_results:
                    return processed_results[:max_results]  # Limit to max_results
                else:
                    self.console.print("[yellow]Warning:[/] No search results found.")
                    return []
            except (requests.RequestException, ConnectionError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.console.print(f"[yellow]Search attempt {attempt+1} failed. Retrying in {wait_time}s...[/]")
                    time.sleep(wait_time)
                else:
                    self.console.print(f"[bold red]Error:[/] Failed to search after {self.max_retries} attempts: {str(e)}")
                    return []
        return []
    
    def generate_summary(self, query: str, results: List[Dict[str, Any]], temperature: float = 0.7) -> str:
        """Generate a summary from search results using the LLM."""
        if not results:
            return "No information was found to summarize."
        
        # Extract and format results with better handling of missing fields
        combined = "\n\n".join([
            f"SOURCE: {r.get('url', 'No URL')}\nTITLE: {r.get('title', 'No Title')}\nCONTENT: {r.get('snippet', 'No content')}" 
            for r in results
        ])
        
        # Create a comprehensive prompt
        prompt = f"""You are a helpful research assistant. Based on the following search results about "{query}", provide a comprehensive, accurate summary that addresses the query directly.

Include key facts, insights, and different perspectives when available. If the information is insufficient or contradictory, acknowledge this in your response.

SEARCH RESULTS:
{combined}

YOUR SUMMARY:"""

        try:
            for attempt in range(self.max_retries):
                try:
                    self.console.print("\n[bold blue]🧠 Generating summary...[/]")
                    response = self.llm_client.text_generation(
                        prompt, 
                        max_new_tokens=500,
                        temperature=temperature,
                        top_p=0.9,
                    )
                    return response.strip()
                except (HfHubHTTPError, Exception) as e:
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        self.console.print(f"[yellow]LLM request failed. Retrying in {wait_time}s...[/]")
                        time.sleep(wait_time)
                    else:
                        raise e
        except Exception as e:
            self.console.print(f"[bold red]Error generating summary:[/] {str(e)}")
            return f"I encountered an error while trying to generate a summary: {str(e)}"
    
    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """Display the search results in a nicely formatted way."""
        if not results:
            self.console.print("\n[yellow]No search results to display.[/]")
            return
            
        self.console.print("\n[bold green]🔗 Top Web Results:[/]")
        
        for i, r in enumerate(results, 1):
            title = r.get("title", "No Title")
            
            # More robust snippet extraction with fallbacks
            snippet = "No summary available."
            if r.get("snippet"):
                snippet = r.get("snippet")
            elif r.get("content"):
                snippet = r.get("content")
            elif r.get("text"):
                snippet = r.get("text")
            elif r.get("body"):
                snippet = r.get("body")
                
            # Truncate very long snippets
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."
                
            url = r.get("url", "No URL provided")
            
            # Wrap text to prevent long lines
            wrapped_snippet = textwrap.fill(snippet, width=100)
            
            panel_content = f"[bold]{title}[/bold]\n\n{wrapped_snippet}\n\n[dim blue]{url}[/dim blue]"
            self.console.print(Panel(panel_content, title=f"Result {i}", border_style="blue"))
    
    def display_summary(self, summary: str) -> None:
        """Display the AI-generated summary."""
        if not summary:
            return
            
        self.console.print("\n[bold green]🧠 AI Assistant Summary:[/]")
        self.console.print(Panel(Markdown(summary), border_style="green"))
    
    def interactive_mode(self):
        """Run the assistant in interactive mode."""
        self.console.print(Panel.fit(
            "[bold]🤖 Enhanced Web Search Assistant[/]\n"
            "Ask any question to search the web and get an AI-summarized response.\n"
            "Type 'exit', 'quit', or use Ctrl+C to exit.",
            title="Welcome",
            border_style="green"
        ))
        
        history = []
        
        while True:
            try:
                query = input("\n🔍 Ask your question (or 'exit'): ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ["exit", "quit"]:
                    self.console.print("[bold green]👋 Goodbye![/]")
                    break
                
                # Save to history
                history.append(query)
                
                # Execute search
                results = self.search_web(query)
                
                # Display results
                self.display_results(results)
                
                # Generate and display summary
                summary = self.generate_summary(query, results)
                self.display_summary(summary)
                
            except KeyboardInterrupt:
                self.console.print("\n[bold green]👋 Goodbye![/]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Error:[/] {str(e)}")
    
    def run_batch_query(self, query: str) -> Dict[str, Any]:
        """Run a single query and return the results and summary."""
        results = self.search_web(query)
        summary = self.generate_summary(query, results)
        
        return {
            "query": query,
            "results": results,
            "summary": summary
        }


def main():
    """Main function to parse arguments and run the assistant."""
    parser = argparse.ArgumentParser(description="Enhanced Web Search Assistant")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", 
                        help="HuggingFace model to use for summarization")
    parser.add_argument("--query", type=str, help="Run a single query and exit")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for LLM generation (0.0-1.0)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Set up debug options if requested
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    assistant = SearchAssistant(model_name=args.model)
    
    if args.query:
        # Run a single query
        result = assistant.run_batch_query(args.query)
        assistant.display_results(result["results"])
        assistant.display_summary(result["summary"])
    else:
        # Run in interactive mode
        assistant.interactive_mode()


if __name__ == "__main__":
    main()