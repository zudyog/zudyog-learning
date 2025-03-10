from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlmodel import Field, Session, SQLModel, create_engine

sqlite_file_name = "test.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


class Usage(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    prompt: str
    temperature: str
    stop: str
    is_openai: bool
    response: str
    input_tokens: str
    output_tokens: str


sqlite_file_name = "test.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=True, connect_args={
                       "check_same_thread": False})


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def create_usage(usage: Usage) -> Usage:
    with Session(engine) as session:
        session.add(usage)
        session.commit()
        session.refresh(usage)
        session.close()
    return usage


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield
