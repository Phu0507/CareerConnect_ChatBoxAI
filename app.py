from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_chat import router as api_router

app = FastAPI()

# ✅ Cho phép các origin frontend được gọi API
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # hoặc ["*"] tạm thời cho mọi origin
    allow_credentials=True,
    allow_methods=["*"],          # GET, POST, OPTIONS, ...
    allow_headers=["*"],          # cho phép mọi header
)

# router của bạn
app.include_router(api_router)
