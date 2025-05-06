import os
import re
import json
import base64
import psycopg2
from io import BytesIO
from psycopg2 import sql
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

# class DBOps:

# def **init**(self, base64\_string=None, metadata=None):

# self.base64\_string = base64\_string

# self.metadata = metadata

# # Azure OpenAI Creds

# self.OPENAI\_API\_KEY = os.getenv("OPENAI\_API\_KEY")

# self.OPENAI\_EMBEDDING\_MODEL\_NAME = os.getenv("OPENAI\_EMBEDDING\_MODEL\_NAME")

# self.OPENAI\_API\_VERSION = os.getenv("OPENAI\_API\_VERSION")

# self.AZURE\_ENDPOINT = os.getenv("AZURE\_ENDPOINT")

# self.model = "text-embedding-ada-002-vector-creation"

# # PostgreSQL DB Creds

# self.db\_name = os.getenv("DB\_NAME")

# self.user = os.getenv("USER")

# self.password = os.getenv("PASSWORD")

# self.host = os.getenv("HOST")

# self.port = os.getenv("PORT")

# def clean\_text(self, text):

# # Remove special characters and extra whitespace

# # #text = text.replace('\n', '')

# cleaned\_text = re.sub(r'\[^\w\s.]', '', text)   # Remove special characters

# cleaned\_text = re.sub(r'.{2,}', ' ', cleaned\_text)

# return cleaned\_text

# def base64\_to\_pdf\_text(self):

# """Decodes Base64 PDF and extracts text with metadata."""

# pdf\_base64\_string = self.base64\_string

# metadata = self.metadata

# if not isinstance(metadata, dict):

# metadata = {}

# try:

# pdf\_data = base64.b64decode(pdf\_base64\_string)

# except Exception as e:

# raise ValueError("Invalid Base64 String") from e

# pdf\_file = BytesIO(pdf\_data)

# pdf\_reader = PdfReader(pdf\_file)

# texts = \[]

# for page\_num, page in enumerate(pdf\_reader.pages, start=1):

# text = page.extract\_text()

# if text:

# cleaned\_text = self.clean\_text(text)

# texts.append({

# "text": cleaned\_text,

# "metadata": {\*\*metadata, "page\_number": page\_num}

# })

# return texts

# def get\_text\_chunks(self, texts\_with\_metadata):

# """Split cleaned texts into chunks while preserving metadata."""

# text\_splitter = RecursiveCharacterTextSplitter(

# separators=\[""],

# chunk\_size=1000,

# chunk\_overlap=200,

# length\_function=len,

# strip\_whitespace=True

# )

# chunks = \[]

# for entry in texts\_with\_metadata:

# chunks\_with\_meta = text\_splitter.create\_documents(

# texts=\[entry\["text"]],

# metadatas=\[entry\["metadata"]]

# )

# chunks.extend(chunks\_with\_meta)

# return chunks

class DBOps:
def **init**(self, base64\_string=None, metadata=None):
self.base64\_string = base64\_string
self.metadata = metadata or {}

```
    # Azure OpenAI Creds
    self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    self.OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
    self.OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
    self.AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    self.model = "text-embedding-ada-002-vector-creation"

    # PostgreSQL DB Creds
    self.db_name = os.getenv("DB_NAME")
    self.user = os.getenv("USER")
    self.password = os.getenv("PASSWORD")
    self.host = os.getenv("HOST")
    self.port = os.getenv("PORT")

def clean_text(self, text: str) -> str:
    """Remove special characters and extra whitespace."""
    cleaned = re.sub(r'[^\w\s.]', '', text)
    cleaned = re.sub(r'\.{2,}', ' ', cleaned)
    return cleaned.strip()

# def base64_to_pdf_text(self) -> dict:
#     """Decode Base64 PDF, extract all pages, stitch them with page markers."""
#     try:
#         pdf_data = base64.b64decode(self.base64_string)
#     except Exception as e:
#         raise ValueError("Invalid Base64 PDF string") from e

#     reader = PdfReader(BytesIO(pdf_data))
#     pages = []
#     for i, page in enumerate(reader.pages, start=1):
#         raw = page.extract_text() or ''
#         cleaned = self.clean_text(raw)
#         # Optional page break marker
#         pages.append(f"\n\n--- page {i} ---\n\n{cleaned}")

#     full_text = "\n".join(pages)
#     return {"text": full_text, "metadata": self.metadata.copy()}

def base64_to_pdf_text(self) -> dict:
    try:
        pdf_data = base64.b64decode(self.base64_string)
    except Exception as e:
        raise ValueError("Invalid Base64 PDF string") from e

    reader = PdfReader(BytesIO(pdf_data))
    pages = []
    full = []
    offsets = []
    cursor = 0
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ''
        cleaned = self.clean_text(raw)
        # marker + page text
        chunk = f"\n\n--- page {i} ---\n\n{cleaned}"
        pages.append(chunk)
        full.append(chunk)
        start = cursor
        cursor += len(chunk)
        end = cursor
        offsets.append((i, start, end))
    self._full_text = ''.join(full)
    self._page_offsets = offsets
    return {"text": self._full_text, "metadata": self.metadata.copy()}



# def get_text_chunks(self, document: dict) -> list:
#     splitter = RecursiveCharacterTextSplitter(
#     separators=[""],       # never split on newlines, only by character
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     strip_whitespace=True
# )
#     return splitter.create_documents(
#         texts=[document["text"]],
#         metadatas=[document["metadata"]])

# def get_text_chunks(self, document: dict) -> list:
#     """Create overlapping chunks across entire PDF text and tag with page numbers."""
#     splitter = RecursiveCharacterTextSplitter(
#         separators=[""],  # no hard splits; pure character-based
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#         strip_whitespace=True
#     )
#     docs = splitter.create_documents(
#         texts=[document["text"]],
#         metadatas=[document["metadata"]]
#     )
#     # Tag each chunk with page numbers found in its content
#     page_pattern = re.compile(r"--- page (\d+) ---")
#     for doc in docs:
#         found = page_pattern.findall(doc.page_content)
#         # Unique, sorted page numbers
#         pages = sorted({int(p) for p in found}) if found else []
#         doc.metadata['page_number'] = pages[0] if pages else None
#     return docs

def get_text_chunks(self, document: dict) -> list:
    splitter = RecursiveCharacterTextSplitter(
        separators=[""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        strip_whitespace=True
    )
    docs = splitter.create_documents(
        texts=[self._full_text],
        metadatas=[document["metadata"]]
    )
    # map each chunk start to a page
    for doc in docs:
        snippet = doc.page_content[:50]  # use first 50 chars
        idx = self._full_text.find(snippet)
        page_num = None
        for p, start, end in self._page_offsets:
            if start <= idx < end:
                page_num = p
                break
        doc.metadata['page_number'] = page_num
    return docs


def create_embeddings(self, text_with_chunks):

    embeddings = AzureOpenAIEmbeddings(
        deployment=self.OPENAI_EMBEDDING_MODEL_NAME,
        openai_api_key=self.OPENAI_API_KEY,
        azure_endpoint=self.AZURE_ENDPOINT,
        openai_api_version=self.OPENAI_API_VERSION,
        openai_api_type="azure",
        chunk_size=1
    )

    chunks_with_embeddings = []
    for entry in text_with_chunks:
        content = entry.page_content
        content_embeddings = embeddings.embed_query(content)

        # Update the Document object with the new embeddings
        entry.metadata['embedding'] = content_embeddings

        chunks_with_embeddings.append(entry)
        
    return chunks_with_embeddings

def insert_documents_to_db(self, documents):
    """
    Inserts a list of Document objects into the PostgreSQL table 'chatbox_knowledge_base'.
    
    Parameters:
    - documents: List of Document objects.
    - conn_params: Dictionary with PostgreSQL connection parameters.
    """
    # Establish the connection
    conn_params = {
    'dbname': self.db_name,
    'user': self.user,
    'password': self.password,
    'host': self.host,
    'port': self.port
    }

    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO chatbox_knowledge_base (client_id, document_id, document_name, page_number, file_metadata, embedding, page_content, status)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """
    
    for doc in documents:
        metadata = doc.metadata
        client_id = metadata.get('client_id')
        document_id = metadata.get('document_id')
        document_name = metadata.get('document_name')
        page_number = metadata.get('page_number')
        file_metadata = json.dumps(metadata.get('file_metadata'))  # Convert to JSON string
        embedding = metadata.get('embedding')
        page_content = doc.page_content
        status = metadata.get('status')

        cursor.execute(insert_query, (client_id, document_id, document_name, page_number, file_metadata, embedding, page_content, status))

    # Commit the transaction
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()
    print("Documents inserted successfully.")

def delete_documents(self, document_id):
            
    conn_params = {
        'dbname': self.db_name,
        'user': self.user,
        'password': self.password,
        'host': self.host,
        'port': self.port
    }

    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    query = sql.SQL("UPDATE chatbox_knowledge_base SET status = 'inactive' WHERE document_id = %s AND status = 'active'")
    cursor.execute(query, (document_id,))

    conn.commit()

    # Check if any row was affected
    if cursor.rowcount > 0:
        message = f"Document with ID {document_id} deleted successfully."
    else:
        message = f"No document found with ID {document_id}."

    cursor.close()
    conn.close()

    return message

def update_documents(self, documents):
    
    conn_params = {
        'dbname': self.db_name,
        'user': self.user,
        'password': self.password,
        'host': self.host,
        'port': self.port
    }

    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    try:
        for doc in documents:
            metadata = doc.metadata
            document_id = str(metadata.get('document_id'))

            update_status_query = """UPDATE chatbox_knowledge_base SET status = 'inactive' WHERE document_id = %s::text"""
            cursor.execute(update_status_query, (document_id,))
        conn.commit()
        self.insert_documents_to_db(documents)
        print(f"Document {document_id} updated successfully")
    except Exception as e:
        conn.rollback()
        print(f"Error Updating documents with Document ID {document_id}: {e}")
    finally:
        cursor.close()
        conn.close()

def fetch_latest_chat_history(self, session_id: str, user_id: str):
    """
    Fetch chat history from PostgreSQL for a specific session_id and user_id.

    :param session_id: The unique session identifier for the chat.
    :param user_id: The unique user identifier.
    :param db_config: Dictionary containing PostgreSQL connection parameters.
    :return: A dictionary containing chat history.
    """
    conn_params = {
        'dbname': self.db_name,
        'user': self.user,
        'password': self.password,
        'host': self.host,
        'port': self.port
    }
    try:
        # Establish database connection
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Fetch chat history for the given session_id and user_id
        query = """
        SELECT memory_data FROM chatbox_user_session 
        WHERE session_id = %s AND user_id = %s 
        ORDER BY session_id DESC 
        LIMIT 1;
        """
        cursor.execute(query, (session_id, user_id))
        result = cursor.fetchone()

        if result:
            return result
        else:
            return {"messages": []}  # Return empty chat history if no record found

    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return None

    finally:
        if conn:
            cursor.close()
            conn.close()
