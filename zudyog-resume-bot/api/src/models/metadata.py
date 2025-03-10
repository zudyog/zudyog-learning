from pydantic import BaseModel


class Metadata(BaseModel):
    document_id: str
    date_uploaded: str
