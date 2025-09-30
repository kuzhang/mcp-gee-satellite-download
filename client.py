import asyncio
import json
from typing import Any, Dict
import os

from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

#-----------------------------------------------------------------------
#Setup the OpenAI model (non-Azure)
#-----------------------------------------------------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to your environment or .env file."
    )

model = ChatOpenAI(
    model=openai_model,
    api_key=openai_api_key,
)

#-----------------------------------------------------------------------
# Define the agent that will use the MCP server
# to answer queries about Satellite image download.
#-----------------------------------------------------------------------
async def main(prompt: str) -> None:

    # Make sure the right path to the server file is passed.
    mcp_server_path = os.path.abspath(
                            os.path.join(os.path.dirname(__file__), 
                                            "server.py"))
    print("GEE MCP server path: ", mcp_server_path)

    # Create the server parameters for the MCP server
    server_params = StdioServerParameters(
        command="python",
        args=[mcp_server_path],
    )
	# Connect to the local MCP server via stdio: `python server.py`
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read,write) as client:
                print("initializing session")
                await client.initialize()
                print("\nloading tools & prompt")
                tools = await load_mcp_tools(client)
                
                print("Ready to download imagery via Google Earth Engine.")
                print("Invoking 'download_satellite_image'...")
                for tool in tools:
                    print("\nTool :", tool.name, '-', tool.description)

                server_prompt = await load_mcp_prompt(client, "get_llm_prompt", arguments={"query": prompt})

                print("\nPrompt loaded :", server_prompt)

                print("\nCreating agent")
                agent=create_react_agent(model,tools)

                print("\nAnswering prompt : ", prompt)
                agent_response = await agent.ainvoke(
                    {"messages": server_prompt})

                return agent_response["messages"][-1].content

        
    except Exception as e:
        print(f"Error: {e}")
        return 'Error'


if __name__ == "__main__":
    response = asyncio.run(main("""
    Please download the satellite image for a bounding box defined by the coordinates 
    [52.526831501, 24.130796795, 52.578119029, 24.181873633]. The desired date range is 2020-01-01 to 2021-01-31.
    The scale is 10 meters. Split the region into smaller tiles if necessary.
    Select the bands B4, B3, B2, B8, B11.
    Filter for less than 20% cloud cover and download 3 image."""))
    print("\n--------------------------------")
    print("\nAgent response :", response)
    print("\n--------------------------------")