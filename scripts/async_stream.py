import asyncio
from src.graph.graph import graph as builder


async def main():
    app = builder.compile()
    async for event in app.astream_events(
        {"messages": [{"role": "user", "content": "Hi, what can you help me with??"}]},
        version="v2",
    ):
        # Filter the events you care about; e.g., token chunks from the chat model
        if event.get("event") == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(chunk, end="", flush=True)


asyncio.run(main())
