from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase_client import supabase

auth_router = APIRouter()

class LoginRequest(BaseModel):
    email: str
    password: str

@auth_router.post("/login")
def login(request: LoginRequest):
    try:
        res = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password,
        })
        if not res.user:
            raise HTTPException(status_code=401, detail="Invalid credentials.")

        # Find restaurant linked to this user
        restaurant_res = (
            supabase.table("restaurants")
            .select("id, name")
            .eq("owner_id", res.user.id)
            .single()
            .execute()
        )

        if not restaurant_res.data:
            raise HTTPException(status_code=404, detail="No restaurant found for this account.")

        return {
            "access_token": res.session.access_token,
            "restaurant_id": restaurant_res.data["id"],
            "restaurant_name": restaurant_res.data["name"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials.")


@auth_router.post("/logout")
def logout():
    supabase.auth.sign_out()
    return {"message": "Logged out"}