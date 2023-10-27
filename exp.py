import chainlit as cl
import os 
import re

from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from chainlit.types import AskFileResponse
from typing import List, Any

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

welcome_message = """Chainlit relevant abstract extractor and key-term summarizer🤖
1. Upload PDF file(s)
2. Ask questions / extract relevant abstracts
"""


def preprocess_file(file : AskFileResponse) -> List[Document]:
    """
    This function is specifically tailored for Windows OS. 
    Other ways of handling temporary files might results in
    a 'denied-permission' error. 
    """
    import tempfile

    if file.type == 'text/plain':
        loader = TextLoader
    elif file.type == 'application/pdf':
        loader = PyPDFLoader

    tmp_fd, tmp_file_name = tempfile.mkstemp()  # make temp. file

    try:
        # Write to the temporary file
        with os.fdopen(tmp_fd, 'wb') as tmpfile:
            tmpfile.write(file.content)
            tmpfile.flush()
        
        ldr = loader(tmp_file_name)  # load temp file
        documents = ldr.load()  # load documents
        docs = text_splitter.split_documents(documents)  # split texts into 1000-char. chunks

        for i, doc in enumerate(docs):
            doc.metadata['source'] = f'source_{i}'  # add sources for each doc. (useful for when&if LM hallucinates)

        return docs
    
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_name):
            os.remove(tmp_file_name)
        
        
def get_docsearch(file : AskFileResponse) -> Any:
    """
    Preprocesses documents, adds them to user session,
    creates a unique namespace for the file. 
    """
    docs = preprocess_file(file)
    
    cl.user_session.set('docs', docs)  # Save data in the user session

    # Create a unique namespace for the file
    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch


@cl.on_chat_start
async def start():
    # Deciding whether the user wants to ask questions / summarize abstracts
    await cl.Message(content='Welcome to the chat!').send()
    choosing_task = True
    while choosing_task: 
        res = await cl.AskUserMessage(content='Type "summarize" for ' + 
                                              'summartizing abstracts or "qa" for answering doc. questions.',
                                      timeout=180).send()
        task = res['content']

        if task not in ('summarize', 'qa'):
            await cl.Message(content=f'Unrecognized task: "{task}".').send()
        else:
            choosing_task = False

    # As long as no files are uploaded, stay idle
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=['text/plain', 'application/pdf'],
            max_size_mb=20,
            timeout=180
        ).send()

    # Right after recieveing the file, start processing it 
    file = files[0]
    msg = cl.Message(content=f'Processing "{file.name}"...')
    await msg.send()

    # Sync func. -> async func., returns original func in a separate thread
    # Useful for running synchronous tasks w/o blocking the event loop
    docsearch = await cl.make_async(get_docsearch)(file)

    # Init. chain
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type='stuff',
        retriever=docsearch.as_retriever(max_tokens_limit=4000),
    )

    # Let the user now that the system is ready
    msg.content = f'"{file.name}" processed. You can now ask questions!'
    await msg.update()

    cl.user_session.set('chain', chain)  # Pass the chain to user session for later acess


# User's message goes through this function and the wrapper
@cl.on_message
async def main(message):
    chain = cl.user_session.get('chain')  # RetirevalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=['FINAL', 'ANSWER']
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])

    answer = res['answer']
    sources = res['sources'].strip() 
    source_elements = []

    # Get the documents from the user session
    docs = cl.user_session.get('docs')
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m['source'] for m in metadatas]

    # In case the model hallucinates, sources can be investigated by the user
    # The code below is not neccessary, just useful for tracing back sources
    if sources:
        found_sources = []

        # Addd the sources to the message
        for source in sources.split(', '):
            source_name = source.strip().replace('.', '')
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            
            text = re.sub('\s{2,}', docs[index].page_content, ' ')  # PDF-files tend to be janky, fix them
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f'\nSources: {", ".join(found_sources)}'
        else:
            answer += '\nNo sources found'

    # If final answer have already been provided, update it with the sources
    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    # Otherwise, just send the final messsage 
    else:
        await cl.Message(content=answer, elements=source_elements).send()