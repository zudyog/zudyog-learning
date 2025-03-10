import os
import json
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from src.core.utils import convert_to_pdf
from src.database import lifespan
from src.models.metadata import Metadata
from src.models.chat_llm import ChatLLM
from src.models.bot_assistant import BotAssistant
from src.document_handler.document_retrieval import DocumentRetrieval
from src.models.query_model import QueryRequest, QueryResponse

app = FastAPI(lifespan=lifespan)

queries = {}


@app.post("/index_texts/")
async def index_texts_endpoint(metadata: str = Form(...), file: UploadFile = File(...)):
    # Parse the metadata string into a dictionary
    metadata_dict = json.loads(metadata)

    # Convert to your metadata model
    metadata_obj = Metadata(**metadata_dict)
    if 'date_uploaded' not in metadata_dict:
        metadata_dict['date_uploaded'] = datetime.now().isoformat()
    metadata_obj = Metadata(**metadata_dict)
    if not metadata_obj.date_uploaded:
        metadata_obj.date_uploaded = datetime.now().isoformat()
    # Ensure the /tmp directory exists
    os.makedirs("/tmp", exist_ok=True)

    pdf_path = f"/tmp/{file.filename}.pdf"
    if not file.filename.lower().endswith('.pdf'):
        convert_to_pdf(file, pdf_path)
    else:
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

    document_retrieval = DocumentRetrieval()
    num_indexed = document_retrieval.index_texts(
        pdf_path, metadata_obj.model_dump())
    return {"indexed_paragraphs": num_indexed}


@app.post("/query_index/")
async def query_index_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Handles user queries, maintains query context across multiple interactions.
    """
    message = request.text
    temperature = request.temperature
    threshold = request.threshold

    # Generate a new query ID if not provided
    if not request.query_id:
        request.query_id = str(uuid.uuid4())

    # Initialize a new query if it doesn't exist
    if request.query_id not in queries:
        queries[request.query_id] = BotAssistant(
            llm=ChatLLM(
                temperature=temperature, model="gpt-4o"
            ),
            verbose=True,
            threshold=threshold,
        )

    print(f"Query ID: {request.query_id}")
    print(f"Queries: {queries.keys()}")

    # Fetch the bot instance for the given query
    bot = queries[request.query_id]

    # Generate response asynchronously
    response = bot.run(message)

    return QueryResponse(response=response, query_id=request.query_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)
