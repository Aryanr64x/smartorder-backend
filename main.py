
from dotenv import load_dotenv


load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.chat_router import chat_router
from routes.auth_router import auth_router
from routes.order_router import order_router
from routes.dashboard_router import dashboard_router
from routes.leads_router import leads_router


app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000", 'https://smartorder-frontend-nine.vercel.app'],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"])

app.include_router(chat_router, prefix='/menu')
app.include_router(auth_router,       prefix="/auth")
app.include_router(order_router,      prefix="/order")
app.include_router(dashboard_router,  prefix="/dashboard")
app.include_router(leads_router,      prefix="/leads")
 

