from src.graph.vectorstores.in_memory import create_in_memory_retriever_tool
import asyncio


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


_, retriever = create_in_memory_retriever_tool()


async def main():
    docs = await retriever.ainvoke("What products does Aetherix Dynamics offer?")
    pretty_print_docs(docs)


if __name__ == "__main__":
    asyncio.run(main())
