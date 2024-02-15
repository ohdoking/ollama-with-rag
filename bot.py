import os

from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chainlit as cl

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Set up Retrieval QA model
# QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-mistral")

prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}

    Helpful Answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt


def load_llm():
    """Load the Language Model."""
    llm = Ollama(
        model="mistral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm


def retrieval_qa_chain(llm, prompt, vectorstore):
    """
   Creates a Retrieval Question-Answering (QA) chain using a given language model, prompt, and database.

   This function initializes a RetrievalQA object with a specific chain type and configurations,
   and returns this QA chain. The retriever is set up to return the top 3 results (k=3).

   Args:
       llm (any): The language model to be used in the RetrievalQA.
       prompt (str): The prompt to be used in the chain type.
       vectorstore (any): The database to be used as the retriever.

   Returns:
       RetrievalQA: The initialized QA chain.
   """
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


def create_retrieval_qa_bot():
    """
    This function creates a retrieval-based question-answering bot.

    Returns:
        RetrievalQA: The retrieval-based question-answering bot.
    """
    vectorstore = Chroma(persist_directory=os.getenv('DB_PATH'), embedding_function=GPT4AllEmbeddings())

    try:
        llm = load_llm()
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

    qa_prompt = (
        set_custom_prompt()
    )  # Assuming this function exists and works as expected

    try:
        qa = retrieval_qa_chain(llm, qa_prompt, vectorstore)  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

    return qa


@cl.on_chat_start
async def start():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    chain = create_retrieval_qa_bot()
    msg = cl.Message(content="Firing up the research info bot...")
    await msg.send()
    msg.content = "Hi, welcome to research info bot by ohdoking. What is your query?"
    msg.author = "ohdoking bot"
    actions = [
        cl.Action(name="upload_file", value="example_value", description="upload file")
    ]
    msg.actions = actions
    await msg.update()
    cl.user_session.set("chain", chain)


async def get_chain_response(chain, message_content):
    """ This function interacts with the chain and returns its response. """
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    # Execute the bot's call method with the given message and callback.
    res = await chain.acall(message_content, callbacks=[cb])
    print(f"response: {res}")
    return res

def process_source_documents(source_documents):
    """ This function processes the source documents and returns a list of text elements. """
    text_elements = []
    for source_idx, source_doc in enumerate(source_documents):
        source_name = f"{os.path.basename(source_doc.metadata['source'])}_{source_idx}"
        # Create the text element referenced in the message
        text_elements.append(
            cl.Text(content=source_doc.page_content.replace('\n', ' '), name=source_name)
        )
    source_names = [text_el.name for text_el in text_elements]
    return text_elements, source_names

@cl.action_callback("upload_file")
async def on_action(action: cl.Action) -> None:
    UPLOADED_FILES: list[String] = []

    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!", accept=["text/csv"]
        ).send()
    # Decode the file
    text_file = files[0]
    text = text_file.content.decode("utf-8")

    UPLOADED_FILES.append(text_file)

    # Let the user know that the system is ready
    await cl.Message(
        content=f"`{text_file.name}` uploaded, it contains {len(text)} characters!"
    ).send()
    await action.remove()


@cl.on_message
async def process_chat_message(message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    chain = cl.user_session.get("chain")
    res = await get_chain_response(chain, message.content)
    answer = res["result"]
    source_documents = res["source_documents"]

    text_elements, source_names = process_source_documents(source_documents)

    if source_names:
        answer += f"\nSources: {', '.join(source_names)}"
    else:
        answer += "\nNo sources found"

    # Send a response back to the user
    await cl.Message(content=answer, elements=text_elements, author="ohdoking bot").send()
