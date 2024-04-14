import os

from crewai import Agent, Task, Process, Crew

from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.utilities import GoogleSerperAPIWrapper

serper_api_key = os.environ.get("SERPER_API_KEY")

search = GoogleSerperAPIWrapper()

search_tool = Tool(
    name="Google Scraper Tool",
    func=search.run,
    description="useful for when you need to ask the agent to search the internet",
)

# Loading Human Tools
human_tools = load_tools(["human"])

# To load GPT-4
api = os.environ.get("OPENAI_API_KEY")

"""
- define agents that are going to research latest AI tools and write a blog about it 
- explorer will use access to internet to get all the latest news
- writer will write drafts 
- critique will provide feedback and make sure that the blog text is engaging and easy to understand
"""
explorer = Agent(
    role="Senior Researcher",
    goal="Find and explore the most exciting projects and companies in the ai and machine learning space in 2024",
    backstory="""You are and Expert strategist that knows how to spot emerging trends and companies in AI, tech and machine learning. 
    You're great at finding interesting, exciting projects on LocalLLama subreddit. You turned scraped data into detailed reports with names
    of most exciting projects an companies in the ai/ml world. ONLY use scraped data from the internet for the report.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
)

writer = Agent(
    role="Senior Technical Writer",
    goal="Write engaging and interesting blog post about latest AI projects using simple, layman vocabulary",
    backstory="""You are an Expert Writer on technical innovation, especially in the field of AI and machine learning. You know how to write in 
    engaging, interesting but simple, straightforward and concise. You know how to present complicated technical terms to general audience in a 
    fun way by using layman words.ONLY use scraped data from the internet for the blog.""",
    verbose=True,
    allow_delegation=True,
)

critic = Agent(
    role="Expert Writing Critic",
    goal="Provide feedback and criticize blog post drafts. Make sure that the tone and writing style is compelling, simple and concise",
    backstory="""You are an Expert at providing feedback to the technical writers. You can tell when a blog text isn't concise,
    simple or engaging enough. You know how to provide helpful feedback that can improve any text. You know how to make sure that text 
    stays technical and insightful by using layman terms.
    """,
    verbose=True,
    allow_delegation=True,
)
