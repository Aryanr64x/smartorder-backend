"""Resolve which restaurant a guest's session belongs to.

The customer app is opened from a QR code carrying `?table=<id>&restaurant=<id>`,
but that URL is guest-editable, so `table_id` is the only trustworthy half — it's
what the QR was physically printed for. Anything scoped to a restaurant (which
menu to retrieve from, who an order is attributed to) resolves through here
rather than believing the `restaurant` query param.
"""
import threading
import time
from functools import lru_cache

from fastapi import HTTPException

from supabase_client import supabase

# Used only when the app is opened directly, with no QR params at all (mirrors
# the frontend's DEFAULT_TABLE_ID fallback). The seeded demo restaurant.
DEFAULT_RESTAURANT_ID = 2


@lru_cache(maxsize=1024)
def resolve_restaurant_id(table_id: int) -> int:
    """Return the restaurant owning `table_id`, raising 404 if no such table.

    Cached because the mapping is effectively immutable: `create_table` sets
    restaurant_id on insert, nothing ever updates it, and Postgres never reuses a
    deleted serial id — so a warm entry can't become a *wrong* answer.

    lru_cache specifically (rather than a hand-rolled dict) for two properties:
    it's thread-safe, which matters because chat_router runs the pipeline in a
    ThreadPoolExecutor, and it does NOT memoise exceptions — so an unknown table
    404s and is retried on the next scan instead of poisoning the cache.
    """
    res = (
        supabase.table("tables")
        .select("restaurant_id")
        .eq("id", table_id)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Unknown table.")
    return rows[0]["restaurant_id"]


def restaurant_for_request(
    table_id: int | None = None,
    restaurant_hint: int | None = None,
) -> int:
    """Pick the restaurant to scope a guest request to.

    `restaurant_hint` is the `?restaurant=` param — a hint only. When we have a
    table we resolve through it and ignore the hint, so editing the URL can't
    surface another restaurant's menu at this table.
    """
    if table_id is not None:
        restaurant_id = resolve_restaurant_id(table_id)
        if restaurant_hint is not None and restaurant_hint != restaurant_id:
            print(
                f"[tenancy] ?restaurant={restaurant_hint} disagrees with table "
                f"{table_id} (owned by {restaurant_id}) — using {restaurant_id}"
            )
        
        return restaurant_id

    # No table to validate against (direct visit / legacy QR). Menus are public,
    # so an unvalidated hint is acceptable here; there's no order to misattribute.
    if restaurant_hint is not None:
        return restaurant_hint

    return DEFAULT_RESTAURANT_ID


# restaurant_id -> (fetched_at, profile). A short TTL rather than lru_cache:
# unlike the table->restaurant mapping, name / meta_data can be edited in
# Supabase, and the waiter should pick that up without a server restart.
_PROFILE_TTL_SECONDS = 300
_profile_cache: dict[int, tuple[float, dict]] = {}
_profile_lock = threading.Lock()


def restaurant_profile(restaurant_id: int) -> dict:
    """Return {'name', 'meta_data'} for the restaurant, for use in LLM prompts.

    Never raises — a guest's chat shouldn't fail because the profile lookup did;
    it falls back to a generic name and no extra info instead.
    """
    now = time.monotonic()
    with _profile_lock:
        hit = _profile_cache.get(restaurant_id)
        if hit and now - hit[0] < _PROFILE_TTL_SECONDS:
            return hit[1]

    try:
        res = (
            supabase.table("restaurants")
            .select("name, meta_data")
            .eq("id", restaurant_id)
            .limit(1)
            .execute()
        )
        row = (res.data or [{}])[0]
    except Exception as e:
        print(f"[tenancy] restaurant_profile({restaurant_id}) failed: {e}")
        row = {}

    profile = {
        "name": (row.get("name") or "").strip() or "our restaurant",
        "meta_data": (row.get("meta_data") or "").strip(),
    }
    if row:
        with _profile_lock:
            _profile_cache[restaurant_id] = (now, profile)
    return profile
