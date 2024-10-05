from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from service.question_answer_service import QuestionAnswerService

app = FastAPI(title="CUSTOM QA BOT", version="1.0")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a single instance of QuestionAnswerService
service = QuestionAnswerService()

@app.get("/", include_in_schema=False)
async def read_root():
    return "CUSTOM QA BOT v1.0"

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    service.load_pdf(content)  # Load the PDF when uploaded
    return JSONResponse(content={"message": "File uploaded successfully."})

@app.post("/answer_question")
async def answer_question(question: str = Form(...)):
    try:
        answer = service.generate_response(question)  # Generate response based on the loaded content
        return JSONResponse(content={"answer": answer})
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)