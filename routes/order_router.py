from fastapi import APIRouter, HTTPException
from supabase_client import supabase
from pydantic import BaseModel
from typing import List

order_router = APIRouter()

class OrderItem(BaseModel):
    id: int
    name: str
    quantity: int

class PlaceOrderRequest(BaseModel):
    items: List[OrderItem]
    restaurant_id: int = 1
    table_id: int = 1

@order_router.post("")
def place_order(request: PlaceOrderRequest):
    if not request.items:
        raise HTTPException(status_code=400, detail="Cart is empty.")

    # 1. Fetch prices from DB for all item ids
    ids = [item.id for item in request.items]
    menu_res = supabase.table("menu").select("id, price").in_("id", ids).execute()
    price_map = {row["id"]: row["price"] for row in menu_res.data}

    # 2. Compute total
    total = sum(
        price_map.get(item.id, 0) * item.quantity
        for item in request.items
    )
    total_qty = sum(item.quantity for item in request.items)

    # 3. Insert into orders
    order_res = supabase.table("orders").insert({
        "restaurant_id": request.restaurant_id,
        "table_id":      request.table_id,
        "total":         total,
        "items":         total_qty,
        "status":        "pending",
    }).execute()

    if not order_res.data:
        raise HTTPException(status_code=500, detail="Failed to create order.")

    order_id = order_res.data[0]["id"]

    # 4. Insert menu_orders rows (one per item×quantity)
    menu_order_rows = []
    for item in request.items:
        for _ in range(item.quantity):
            menu_order_rows.append({
                "order_id": order_id,
                "menu_id":  item.id,
            })

    supabase.table("menu_orders").insert(menu_order_rows).execute()

    print(f"[order] ✅ Order #{order_id} placed — {total_qty} items, total ₹{total}")

    return {"order_id": order_id, "total": total, "status": "pending"}