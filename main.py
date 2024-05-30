from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import crud, models, schemas
from .database import SessionLocal, engine
from .ml.text_search import TextSearch

app = FastAPI()

models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db=db, user=user)

@app.post("/records/", response_model=schemas.Record)
def create_record(record: schemas.RecordCreate, db: Session = Depends(get_db)):
    user_id = record.user_id
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return crud.create_record(db=db, record=record, user_id=user_id)

@app.get("/users/{user_id}/similar_records/")
def find_similar_records(user_id: int, db: Session = Depends(get_db)):
    records = db.query(models.Record).filter(models.Record.user_id == user_id).all()
    embeddings = [record.embedding for record in records]
    search_model = TextSearch()
    search_model.update_corpus(records)
    search_model.train_model()

    similarities = []
    for i, embedding in enumerate(embeddings):
        for j in range(i + 1, len(embeddings)):
            sim = search_model.cosine_similarity(embedding, embeddings[j])
            similarities.append((i, j, sim))
    return similarities
