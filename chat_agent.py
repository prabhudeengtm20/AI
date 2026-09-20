import os

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.openai import OpenAI

import chainlit as cl
from chainlit.input_widget import Select, TextInput

from index_wikipages import create_index
from utils import get_apikey


index = None
agent = None


@cl.on_chat_start
async def on_chat_start():
    global index
    settings = await cl.ChatSettings(
        [
            Select(
                id="MODEL",
                label="OpenAI Model Select",
                values=["gpt-4o-mini"],
                initial_index=0,
            ),
            TextInput(
                id="WIKI_QUERY",
                label="Wikipedia Index Request",
                initial_value="Please index: Paris, Lagos, London",
            ),
        ]
    ).send()


def wikisearch_engine(index):
    return index.as_query_engine(
        response_mode="compact",
        verbose=True,
        similarity_top_k=10,
    )


def create_react_agent(MODEL):
    query_engine_tools = [
        QueryEngineTool(
            query_engine=wikisearch_engine(index),
            metadata=ToolMetadata(
                name="wikipedia_search",
                description="Useful for searching information about indexed Wikipedia pages.",
            ),
        )
    ]

    llm = OpenAI(model=MODEL, api_key=get_apikey())
    agent = ReActAgent(tools=query_engine_tools, llm=llm)
    return agent


@cl.on_settings_update
async def setup_agent(settings):
    global agent
    global index
    query = settings["WIKI_QUERY"]
    index = create_index(query)

    print("on_settings_update", settings)
    MODEL = settings["MODEL"]
    agent = create_react_agent(MODEL)
    await cl.Message(
        author="Agent", content=f'Wikipage(s) "{query}" successfully indexed'
    ).send()


@cl.on_message
async def main(message: cl.Message):
    global agent
    if agent:
        response = await agent.run(message.content)
        await cl.Message(author="Agent", content=str(response)).send()
    else:
        await cl.Message(
            author="Agent",
            content="Please submit or verify your Wikipedia indexing settings panel configuration first!",
        ).send()
