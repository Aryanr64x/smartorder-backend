from fastapi import APIRouter, HTTPException
from supabase_client import supabase
from pydantic import BaseModel
import re

leads_router = APIRouter()

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

class LeadRequest(BaseModel):
    name: str
    phone: str
    email: str
    restaurant_name: str
    city: str

@leads_router.post("")
def create_lead(request: LeadRequest):
    # Basic normalisation / validation so we don't store empty rows we then
    # have to chase up manually.
    name            = request.name.strip()
    phone           = request.phone.strip()
    email           = request.email.strip().lower()
    restaurant_name = request.restaurant_name.strip()
    city            = request.city.strip()

    if not (name and phone and restaurant_name and city):
        raise HTTPException(status_code=400, detail="Please fill in all fields.")

    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="That email doesn't look right.")

    try:
        lead_res = supabase.table("leads").insert({
            "name":            name,
            "phone":           phone,
            "email":           email,
            "restaurant_name": restaurant_name,
            "city":            city,
            "status":          "new",
        }).execute()
    except Exception as e:
        print(f"[leads] ❌ Failed to insert lead: {e}")
        raise HTTPException(status_code=500, detail="Could not save your details. Please try again.")

    if not lead_res.data:
        raise HTTPException(status_code=500, detail="Could not save your details. Please try again.")

    lead_id = lead_res.data[0]["id"]
    print(f"[leads] ✅ New lead #{lead_id} — {restaurant_name} ({city}) · {name} · {phone} · {email}")

    return {"lead_id": lead_id, "message": "Thanks! We'll reach out to you soon."}
