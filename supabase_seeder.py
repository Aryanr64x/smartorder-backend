"""Re-seed the Supabase `menu` table from menu.json.

The raw menu.json only has {name, description, price(nested or flat)}. The DB
table also needs `food_type` (enum: dessert/maincourse/starters/beverages) and
`vegnonveg` (bool: True=veg). Everything in menu.json is a soup/salad/starter,
so food_type is "starters" for all rows. veg/nonveg is derived from the name.

Run:  ./venv/Scripts/python.exe supabase_seeder.py
"""
import json
from supabase_client import supabase

# Menu rows FK to restaurants.id — the only restaurant present is id=2.
RESTAURANT_ID = 2

# Items that contain meat/seafood -> vegnonveg = False (non-veg). Everything
# else (paneer, mushroom, corn, veg soups/salads/chaat) is veg.
NONVEG_NAMES = {
    "Murg Ke Chaamp", "Murg Kasturi", "Murg Malai Kebab", "Tandoori Murg",
    "Gosht Seekh Kebab", "Smoked Mutton Chops", "Roti Pe Boti", "Galouti Kebab",
    "Lehsooni Fish Tikka", "Tawa Fish", "Bhattiwala Nimbu Mirchi Jhinga",
    "Hakka Chilli Chicken", "Devils Chicken", "Drums Of Heaven", "Mongolian Fish",
    "Schezwan Pepper Prawn", "Butter Garlic Prawns", "Chilli Prawns",
}


def flat_price(price):
    """menu.json price is either a number or {veg, chicken}. Use the veg price
    when it's a dict (those items are seeded as their veg variant)."""
    if isinstance(price, dict):
        return price.get("veg") or next(iter(price.values()))
    return price


def main():
    with open("menu.json", encoding="utf-8") as f:
        items = json.load(f)

    # Clear any existing rows so re-runs are idempotent (id >= 0 matches all).
    supabase.table("menu").delete().gte("id", 0).execute()

    rows = []
    for it in items:
        rows.append({
            "name": it["name"],
            "description": it.get("description", "") or "",
            "food_type": "starters",
            "vegnonveg": it["name"] not in NONVEG_NAMES,
            "price": flat_price(it["price"]),
            "restaurant_id": RESTAURANT_ID,
        })

    resp = supabase.table("menu").insert(rows).execute()
    inserted = resp.data or []
    veg = sum(1 for r in inserted if r["vegnonveg"])
    print(f"Inserted {len(inserted)} rows  |  veg={veg}  nonveg={len(inserted) - veg}")
    for r in inserted[:5]:
        print(" ", {k: r[k] for k in ("id", "name", "food_type", "vegnonveg", "price")})


if __name__ == "__main__":
    main()
