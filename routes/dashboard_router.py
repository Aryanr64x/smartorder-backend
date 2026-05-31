from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase_client import supabase

dashboard_router = APIRouter()
bearer = HTTPBearer()

def get_restaurant_id(credentials: HTTPAuthorizationCredentials = Depends(bearer)) -> int:
    """Verify JWT and return restaurant_id for the authenticated user."""
    token = credentials.credentials
    try:
        # Verify token with Supabase — returns user if valid
        user_res = supabase.auth.get_user(token)
        if not user_res.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")

        restaurant_res = (
            supabase.table("restaurants")
            .select("id")
            .eq("owner_id", user_res.user.id)
            .single()
            .execute()
        )
        if not restaurant_res.data:
            raise HTTPException(status_code=404, detail="No restaurant found.")

        return restaurant_res.data["id"]
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


@dashboard_router.get("/orders")
def get_orders(restaurant_id: int = Depends(get_restaurant_id)):
    """Fetch all orders for this restaurant with their menu items."""

    # Get orders
    orders_res = (
        supabase.table("orders")
        .select("*")
        .eq("restaurant_id", restaurant_id)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    orders = orders_res.data or []

    if not orders:
        return {"orders": []}

    # For each order fetch its menu items via menu_orders join
    order_ids = [o["id"] for o in orders]
    menu_orders_res = (
        supabase.table("menu_orders")
        .select("order_id, menu_id, menu(name, price)")
        .in_("order_id", order_ids)
        .execute()
    )

    # Group menu items by order_id
    items_by_order: dict = {}
    for row in (menu_orders_res.data or []):
        oid = row["order_id"]
        if oid not in items_by_order:
            items_by_order[oid] = []
        items_by_order[oid].append({
            "menu_id": row["menu_id"],
            "name": row["menu"]["name"] if row.get("menu") else "Unknown",
            "price": row["menu"]["price"] if row.get("menu") else 0,
        })

    # Attach items to orders
    for order in orders:
        order["menu_items"] = items_by_order.get(order["id"], [])

    return {"orders": orders}


@dashboard_router.patch("/orders/{order_id}/status")
def update_order_status(
    order_id: int,
    body: dict,
    restaurant_id: int = Depends(get_restaurant_id)
):
    """Update order status: pending → preparing → done"""
    new_status = body.get("status")
    if new_status not in ("pending", "preparing", "done"):
        raise HTTPException(status_code=400, detail="Invalid status.")

    # Verify order belongs to this restaurant
    order_res = (
        supabase.table("orders")
        .select("id")
        .eq("id", order_id)
        .eq("restaurant_id", restaurant_id)
        .single()
        .execute()
    )
    if not order_res.data:
        raise HTTPException(status_code=404, detail="Order not found.")

    supabase.table("orders").update({"status": new_status}).eq("id", order_id).execute()
    return {"order_id": order_id, "status": new_status}