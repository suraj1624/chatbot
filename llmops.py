import os
import re
import json
import uvicorn
import psycopg2
from psycopg2 import sql
from typing import Dict, Any
from dotenv import load\_dotenv
from pydantic import BaseModel
from openai import AzureOpenAI
from datetime import date, datetime
from database\_operations import DBOps
from langchain.chains import LLMChain
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.text\_splitter import RecursiveCharacterTextSplitter
from langchain\_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.output\_parsers import StructuredOutputParser, ResponseSchema

app = FastAPI()

# Adding CORS middleware

app.add\_middleware(
CORSMiddleware,
allow\_origins=\["*"],  # Allow all origins for testing; use specific origins in production
allow\_credentials=True,
allow\_methods=\["*"],
allow\_headers=\["\*"],
)

load\_dotenv()

# Pydantic Class For Uploading and Pre-processing PDF

class UploadedPDF(BaseModel):
base64\_string: str
metadata: Dict\[str, Any]

# Pydantic Class for user query

class UserQuery(BaseModel):
client\_id: str
user\_type: str
session\_id: str
user\_id: str
query: str

class ChatHistoryParams(BaseModel):
session\_id: str
user\_id: str

# PostgreSQL Connection Settings

DB\_NAME = os.getenv("DB\_NAME")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")

# Azure OpenAI settings

OPENAI\_API\_KEY = os.getenv("OPENAI\_API\_KEY")
OPENAI\_DEPLOYMENT\_NAME = os.getenv("OPENAI\_DEPLOYMENT\_NAME")
OPENAI\_EMBEDDING\_MODEL\_NAME = os.getenv("OPENAI\_EMBEDDING\_MODEL\_NAME")
MODEL\_NAME = os.getenv("MODEL\_NAME")
OPENAI\_API\_VERSION = os.getenv("OPENAI\_API\_VERSION")
AZURE\_ENDPOINT = os.getenv("AZURE\_ENDPOINT")

detector\_client = AzureOpenAI(
azure\_endpoint=AZURE\_ENDPOINT,
api\_key=OPENAI\_API\_KEY,
api\_version="2024-05-01-preview"
)

detector\_deployment\_name = OPENAI\_DEPLOYMENT\_NAME

conn\_params = {
'dbname': DB\_NAME,
'user': USER,
'password': PASSWORD,
'host': HOST,
'port': PORT
}

def get\_db\_connection(conn\_params):
try:
conn = psycopg2.connect(\*\*conn\_params)
return conn
except Exception as e:
print(f"Error connecting to the database: {e}")
raise HTTPException(status\_code=500, detail="Database connection error.")

def exit\_response():
return "Thank you for chatting with us! Have a great day!"

def detect\_intent(query):
"""Detect the intent of the user query."""
print(f"detect\_intent called with query: {query}")

```
messages = [
    {
        "role": "system",
        "content": (
            "You are an intent detection assistant. Detect the intent of the user query based on these defined categories:\n"
            "1. Official_Query: Queries related to official matters, requests, or tasks.\n"
            "2. Exit_Query: Queries indicating the user wants to exit, leave, or terminate the interaction.\n"
            "Examples of Exit_Query include: 'bye', 'exit', 'goodbye', 'end', 'I'm done', 'stop', 'terminate', "
            "'close the session', 'thank you, goodbye', etc.\n"
            "Strictly respond with only the name of the detected intent (Official_Query or Exit_Query)."
        )
    },
    {
        "role": "user",
        "content": query
    }
]

response = detector_client.chat.completions.create(
    model=detector_deployment_name,
    messages=messages
)

intent = response.choices[0].message.content.strip().lower() if response.choices[0].message.content else "unknown_intent"
print(f"Intent detected: {intent}")
return intent
```

# Fetch Documents Using PostgreSQL

def fetch\_documents(query, client\_id, top\_k=10):

```
model: str = "text-embedding-ada-002-vector-creation"

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    deployment=OPENAI_EMBEDDING_MODEL_NAME,
    openai_api_key=OPENAI_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_type="azure",
    openai_api_version=OPENAI_API_VERSION,
    chunk_size=1
)

query_embeddings = embeddings.embed_query(query)

conn = get_db_connection(conn_params)
cursor = conn.cursor()

cursor.execute("""
SELECT client_id, document_id, document_name, page_number, file_metadata, page_content, status, embedding <=> %s::vector AS similarity_score
FROM chatbox_knowledge_base
WHERE client_id = %s AND status = 'active'
ORDER BY similarity_score
LIMIT %s
""", (query_embeddings, client_id, top_k))

results = cursor.fetchall()

documents = []
if len(results) > 0:
    for row in results:
        documents.append({
            "client_id": row[0],
            "document_id": row[1],
            "document_name": row[2],
            "page_number": row[3],
            "file_metadata": row[4],
            "page_content": row[5],
            "similarity_score": row[7]
        })
else:
    print("No Documents Found")

cursor.close()
conn.close()

return documents
```

def structured\_chat\_history(chat\_history):
structured\_history = \[]

````
for message in chat_history.get("messages", []):
    if message["type"] == "human":
        structured_history.append({"role": "User", "message": message["content"]})
    elif message["type"] == "ai":
        ai_content = message["content"].strip("```json").strip()
        structured_history.append({"role": "AI", "message": ai_content})

return structured_history
````

# formatted\_chat = extract\_chat\_history(chat\_history)

# for entry in formatted\_chat:

# print(f"{entry\['role']}: {entry\['message']}\n")

# @app.post('/end-session')

# def terminate\_session(session\_id: SessionID):

# end\_session(session\_id=session\_id.session\_id)

# return {"message": f"Session {session\_id.session\_id} has been successfully terminated."}

@app.put("/update-documents")
def update\_document(pdf\_data: UploadedPDF):

```
base64_string = pdf_data.base64_string
metadata = pdf_data.metadata

dbops = DBOps(base64_string, metadata)

# Pre-process the incoming PDF
raw_text = dbops.base64_to_pdf_text()
print("**********RAW DATA******************")
print(raw_text)
text_with_chunks = dbops.get_text_chunks(raw_text)
print("********************CHUNKS*************************")
print(text_with_chunks)
chunks_with_embeddings = dbops.create_embeddings(text_with_chunks)

# Update the document in the database
dbops.update_documents(chunks_with_embeddings)

return {"message": "PDF Update completed successfully."}
```

@app.delete("/delete-document/{document\_id}")
def delete\_document(document\_id: str = Path(..., description="The ID of the document to delete")):
dbops = DBOps()

```
# Delete the document from the database
delete_request = dbops.delete_documents(document_id)

if delete_request:
    return {"message": delete_request}
else:
    raise HTTPException(status_code=404, detail="Document not found or could not be deleted")
```

@app.post('/chat\_history')
def get\_chat\_history(chat\_history\_params: ChatHistoryParams):

```
session_id = chat_history_params.session_id
user_id = chat_history_params.user_id

dbops = DBOps()

# Fetch the chat history from the database
chat_history = dbops.fetch_latest_chat_history(session_id, user_id)

try:
    if chat_history:
        formated_chat_history = structured_chat_history(chat_history[0])
        return formated_chat_history
    else:
        return {"message": "No chat history found for the specified session."}
except Exception as e:
    return {"message": f"Error fetching chat history: {e}"}
```

@app.post('/preprocess-pdf')
def preprocess\_pdf(pdf\_data: UploadedPDF):

```
base64_string = pdf_data.base64_string
metadata = pdf_data.metadata

dbops = DBOps(base64_string, metadata)
print("****************DBOPS********************")
print(dbops)
# Pre-process the incoming PDF
raw_text = dbops.base64_to_pdf_text()
print("**********RAW DATA******************")
print(raw_text)
text_with_chunks = dbops.get_text_chunks(raw_text)
print("********************CHUNKS*************************")
print(text_with_chunks)
chunks_with_embeddings = dbops.create_embeddings(text_with_chunks)

# Insert the pre-processed PDF into the database
dbops.insert_documents_to_db(chunks_with_embeddings)

return {"message": "PDF pre-processing completed successfully."}

@app.post('/chatbox-response')
def generate\_response(user\_query: UserQuery):

client_id = user_query.client_id
type_of_user = user_query.user_type
session_id = user_query.session_id
user_id = user_query.user_id
query = user_query.query

conn = get_db_connection(conn_params)
cursor = conn.cursor()

cursor.execute("SELECT memory_data FROM chatbox_user_session WHERE session_id = %s", (session_id,))
session_memory_data = cursor.fetchone()

if session_memory_data:
    memory_data = session_memory_data[0]
    print("Memory Data:- ",memory_data)

    # Re-initialize the ConversationBufferMemory object with the correct configuration
    session_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="query")

    # Load the chat history from the database into the memory object
    if isinstance(memory_data, str):
        memory_data = json.loads(memory_data)  # Parse JSON string if necessary
    
    # Convert dictionaries back into LangChain message objects
    messages = []
    for msg_dict in memory_data.get("messages", []):
        print("Message Dict:- ",msg_dict)
        if msg_dict["type"] == "human":
            messages.append(HumanMessage(content=msg_dict["content"]))
        elif msg_dict["type"] == "ai":
            messages.append(AIMessage(content=msg_dict["content"]))
    
    session_memory.chat_memory.messages = messages

else:
    # Initialize new session memory if it doesn't exist
    session_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="query")
    print("Messages in session memory: ", session_memory.chat_memory.messages)

    # Insert new session into the database
    memory_messages = [msg.dict() for msg in session_memory.chat_memory.messages]
    json_memory_messages = json.dumps({"messages": memory_messages})
    cursor.execute("INSERT INTO chatbox_user_session (user_id, session_id, memory_data) VALUES (%s, %s, %s)", (user_id, session_id, json_memory_messages))
    conn.commit()

user_query_intent = detect_intent(query)

if user_query_intent.lower() == 'exit_query':
    #end_session(session_id=session_id)
    return {
        "session_id": session_id, 
        "response": {
            "content": [
                {
                    "heading": "",
                    "content_points": [exit_response()]
                }
            ],
            "References":[]
        }
    }

azure_llm = AzureChatOpenAI(
    azure_deployment=OPENAI_DEPLOYMENT_NAME,
    openai_api_key=OPENAI_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_type="azure",
    openai_api_version=OPENAI_API_VERSION,
    model="gpt-4o-mini",
    temperature=0.5
)

response_schemas = [
    ResponseSchema(
        name="content",
        description=(
            "A list of dictionaries where each dictionary represents a section of the content. "
            "Each dictionary should have the following keys: "
            "- 'heading': The main heading as a string, can be empty in case of a greeting string"
            "- 'content_points': A list of bullet points related to the heading. or a list of greeting message (e.g. Hello! How can I assist you today?)"
        )
    ),
    ResponseSchema(
        name="References",
        description=(
            "A list of references in the form of 'document_name/page=Numberofthepage'. "
            "Each entry in the list should represent a document along with its corresponding page number, "
            "formatted as 'document_name/page=Numberofthepage'."
        )    
    )
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

prompt_template = """You are an expert assistant. Your task is to respond only using the information provided in the context and follow these rules:
              
              1. If the user's query can be answered using the context:
                - Provide the answer in a bulleted format if appropriate.
                - Provide the page numbers and document's name for each piece of information used
              2. If the user's query is unrelated to the context:
                - Respond with: "I'm sorry, I can only answer questions based on the provided information."
              3. If the user greets you (e.g., says "Hello", "Hi", "Good morning", "Good evening", "Hey there", "how you doing"):
                - Respond with a polite greeting, (e.g. "Hello! How can I assist you today?").
                - make sure no headings are generated
             
             *STRICTLY - You are not allowed to answer from previous conversation history provided to you. It is provided for logging only, not for answering the question.

            [Do not use following chat history while answering user's query.]     
            Previous Conversation History:
            {chat_history}


            Context: {documents}
            Question: {query}

            Format of the Response
            Generate Response
            Page Numbers (All the possible page number from which the response is created)
            Document Name (Name of the Documents from which the response is created)

            {format_instructions}
"""

prompt = PromptTemplate(input_variables=["chat_history", "documents", "query"],
                        partial_variables={"format_instructions": format_instructions},
                        template=prompt_template)

#llm_chain = LLMChain(llm=azure_llm, prompt=prompt, memory=session_memory)
llm_chain = LLMChain(llm=azure_llm, prompt=prompt, memory=session_memory)

documents = fetch_documents(query, client_id=client_id)

# Getting the current Date
current_date = date.today()

user_type_documents = {}

for doc in documents:
    client_id = doc["client_id"]
    document_name = doc["document_name"]
    page_number = doc["page_number"]
    file_metadata = doc["file_metadata"]
    user_type = file_metadata.get("user_type", "Unknown")
    start_date = file_metadata.get("start_date", None)
    expiration_date = file_metadata.get("expiration_date", None)
    doc_content = doc["page_content"]

    # Skiping the documents without a start date and an expiration date
    if not expiration_date or not start_date:
        continue

    # Adjusting the date format
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()
    except ValueError:
        # Skipping the documents with invalid date format
        continue

    # Skipping documents who does not lay between the start date and the expiration date.
    if not (start_date <= current_date <= expiration_date):
        continue

    # Handle multiple user types
    user_types = [ut.strip() for ut in user_type.split(",")]

    # Format the document text
    doc_text = f"Document Name: {document_name}\nPage Number: {page_number}\nUser Type: {user_type}\nExpiration Date: {expiration_date}\nContent: {doc_content}\n\n" 

    for ut in user_types:
        # Convert user type to a standardized key format
        ut_key = ut.lower().replace(" ", "_").replace("/", "_")
    
        # Initialize entry if it doesn't exist
        if ut_key not in user_type_documents:
            user_type_documents[ut_key] = ""
        
        # Add document to appropriate user type
        user_type_documents[ut_key] += doc_text

# Standardize the incoming user type for dictionary lookup
standardized_user_type = type_of_user.lower().replace(" ", "_").replace("/", "_")

# Get documents for the user type or return None if not found
documents_text = user_type_documents.get(standardized_user_type, None)
print("************Context***********************")
print(documents_text)
try:
    response = llm_chain.invoke(input={"query": query, "documents": documents_text})

    # Update the memory_data in the database with the latest chat history
    updated_memory_messages = [msg.dict() for msg in session_memory.chat_memory.messages]
    updated_memory_data = json.dumps({"messages": updated_memory_messages})
    cursor.execute("UPDATE chatbox_user_session SET memory_data = %s WHERE session_id = %s", (updated_memory_data, session_id))
    conn.commit()

    # Process and return the response
    if "```json" in response["text"]:
        cleaned_response = response["text"].strip().strip('```json').strip()
        cleaned_response = cleaned_response.replace('\n', '').replace('\t', '')
        json_object = json.loads(cleaned_response)
        return {"session_id": session_id, "response": json_object}
    else:
        cleaned_response = response["text"]
        cleaned_response = cleaned_response.replace('\n', '').replace('\t', '')
        return {
            "session_id": session_id, 
            "response": {
                "content": [
                    {
                        "heading": "",
                        "content_points": [cleaned_response]
                    }
                ],
                "References":[]
            }
        }  
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    raise HTTPException(status_code=500, detail="Error processing the response")