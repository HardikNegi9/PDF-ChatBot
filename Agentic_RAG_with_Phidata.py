import typer
from typing import Optional
from rich.prompt import Prompt

from phi.agent import Agent
# from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.vectordb.lancedb import LanceDb
from phi.vectordb.search import SearchType
from phi.model.groq import Groq
from phi.embedder.huggingface import HuggingfaceCustomEmbedder

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables for the phi framework
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# LanceDB Vector DB
vector_db = LanceDb(
    table_name="reserach",
    uri="/tmp/lancedb",
    search_type=SearchType.keyword,
    embedder=HuggingfaceCustomEmbedder(),
)

# Knowledge Base
# knowledge_base = PDFUrlKnowledgeBase(
#     urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#     vector_db=vector_db,
# )

knowledge_base = PDFKnowledgeBase(
    path="12. Fine-Tuning Generation Models _ Hands-On Large Language Models.pdf",
    # Table name: ai.pdf_documents
    vector_db=vector_db,
    reader=PDFReader(chunk=True),
)



# Load knowledge base (comment this out after the first run)
if not os.path.exists("/tmp/lancedb"):
    print("Loading the knowledge base for the first time...")
    knowledge_base.load(recreate=True)
else:
    print("Knowledge base already loaded.")

# Define the agent
def lancedb_agent(user: str = "user"):
    run_id: Optional[str] = None

    agent = Agent(
        run_id=run_id,
        user_id=user,
        knowledge=knowledge_base,
        show_tool_calls=True,
        debug_mode=True,
        model=Groq(id="gemma2-9b-it"),
    )

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message.lower() in ("exit", "bye"):
            print("Exiting... Goodbye!")
            break
        response = agent.print_response(message)
        print(f"Agent Response: {response}")


if __name__ == "__main__":
    typer.run(lancedb_agent)
