from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from app.ocr import load_image
from app.model import model, tokenizer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.ocr import ocr_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc frontend domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ocr/parse")
# async def ocr_parse(file: UploadFile = File(...)):
#     image = await file.read()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     pixel_values = load_image(image_file=image, input_size=448, max_num=4).to(torch.bfloat16).to(device)
#
#     question = '<image>\nMô tả hình ảnh một cách chi tiết trả về dạng markdown.'
#     response = model.chat(
#         tokenizer, pixel_values, question,
#         generation_config=dict(max_new_tokens=512, do_sample=False, num_beams=3, repetition_penalty=3.5)
#     )
#     return {"text": response}

async def ocr_parse(file: UploadFile = File(...)):
    image = await file.read()
    response = ocr_image(image)
    return {"text": response}

# Mount thư mục static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Trả về file HTML khi truy cập /dangky
@app.get("/dangkyxenhapkho")
def serve_html():
    return FileResponse("static/DangKyXeNhapKho.html")