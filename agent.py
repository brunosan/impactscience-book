from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

# The language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Let's test it out!
agent.run("The current folder contains an old version of my book. The latest text is here 'https://gist.githubusercontent.com/brunosan/3026db3bcf562ef738e256a1310f8d79/raw/b4884136e6367c543bb5807ceaf7f09923cbaff6/Impact-Science_book_v8.9.txt'. Update the text, and add a text file with an assessment of the book, strenght, weaknesses, section to improve/add/remove, section to add references, ...")
