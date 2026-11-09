"""
Extract Brand Candidates from productDisplayName (name in catalog.parquet).
Saves brand candidates appearing in >= 20 products to src/brands.json.
Exports top 100 brand candidates for user review.
"""
import pandas as pd
from collections import Counter
import json, re
from pathlib import Path

def extract_brand_candidates():
    catalog_path = Path("data/catalog/catalog.parquet")
    df = pd.read_parquet(catalog_path)
    in_scope = df[df["in_scope"] == True]
    names = in_scope["name"].dropna().astype(str).tolist()

    NON_BRAND_TOKEN2 = {
        # Genders
        "men", "women", "unisex", "boys", "girls", "mens", "womens", "kids", "boy", "girl",
        "men's", "women's", "boys'", "girls'", "woman", "man",
        # Article types
        "tshirt", "t-shirt", "shirt", "shirts", "tee", "tees", "shoes", "shoe", "watch", "watches",
        "jeans", "track", "pant", "pants", "trousers", "trouser", "shorts", "short", "sandal", "sandals",
        "bag", "bags", "handbag", "handbags", "backpack", "backpacks", "wallet", "wallets", "belt", "belts",
        "socks", "sock", "top", "tops", "kurta", "kurtas", "kurti", "kurtis", "dress", "dresses",
        "skirt", "skirts", "briefs", "brief", "boxers", "boxer", "trunk", "sunglasses", "sunglass",
        "flats", "flat", "heels", "heel", "cap", "caps", "hat", "hats", "scarf", "scarves", "stole",
        "perfume", "deodorant", "flip", "flops", "clutch", "clutches", "leggings", "jacket", "jackets",
        "sweater", "sweaters", "sweatshirt", "sweatshirts", "innerwear", "vest", "vests", "blazer", "blazers",
        "muffler", "mufflers", "tie", "ties", "cufflinks", "ring", "bangle", "bracelet", "pendant", "necklace",
        # Colors
        "black", "white", "blue", "red", "green", "grey", "gray", "yellow", "pink", "purple", "brown",
        "navy", "gold", "silver", "beige", "orange", "olive", "maroon", "cream", "charcoal", "khaki",
        "teal", "magenta", "bronze", "copper", "turquoise", "burgundy", "peach", "lavender", "tan",
        "rust", "skin", "metallic", "coffee", "sea", "rose", "taupe", "mauve", "lime", "nude",
        # Modifiers / Patterns
        "solid", "printed", "striped", "check", "checked", "graphic", "pack", "set", "pair", "casual",
        "sports", "formal", "party", "ethnic", "round", "crew", "v-neck", "fit", "slim", "regular",
        "classic", "essential", "leather", "cotton", "denim", "canvas", "poly", "dry", "cool", "active",
        "polka", "sleeveless", "full", "half", "short", "long", "analog", "digital", "chronograph"
    }

    # Major single-word brands where 2nd word might look like a name but is just a sub-line/model
    MAJOR_SINGLE_BRANDS = {
        "nike", "adidas", "puma", "fila", "fastrack", "reebok", "casio", "titan", "catwalk",
        "baggit", "fabindia", "wrangler", "arrow", "lee", "myntra", "proline", "wildcraft",
        "lotto", "rocia", "basics", "fossil", "timex", "sonata", "maxima", "fanaang", "murcia",
        "jockey", "scullers", "w", "gini", "spykar", "converse", "crocs", "bata", "woodland"
    }

    # Known 2-word brands to always combine
    KNOWN_2WORD_BRANDS = {
        "united colors", "french connection", "peter england", "lino perros", "jealous 21",
        "flying machine", "mark taylor", "mother earth", "indigo nation", "red tape", "numero uno",
        "carlton london", "john miller", "john players", "franco leone", "park avenue", "lee cooper",
        "status quo", "allen solly", "royal diadem", "arrow sport", "urban yoga", "american tourister",
        "wills lifestyle", "spice art", "red rose", "red chief", "little miss", "loco show",
        "sweet dreams", "classic polo", "provogue men", "police men", "ben 10", "roadster men"
    }

    # Canonical brand normalizer
    def canonical_brand(raw_candidate: str) -> str:
        c_lower = raw_candidate.lower()
        if c_lower in {"united colors", "united", "ucb", "benetton"}:
            return "United Colors of Benetton"
        if c_lower in {"u.s. polo", "us polo", "u.s polo"}:
            return "U.S. Polo Assn."
        if c_lower in {"gini and", "gini", "gini & jony", "gini and jony"}:
            return "Gini and Jony"
        if c_lower in {"reid &", "reid", "reid & taylor"}:
            return "Reid & Taylor"
        if c_lower in {"sdl by", "sdl", "sdl by sweet dreams"}:
            return "SDL by Sweet Dreams"
        if c_lower in {"scullers for", "scullers"}:
            return "Scullers"
        if c_lower in {"adidas originals", "adidas"}:
            return "ADIDAS"
        if c_lower in {"casio"}:
            return "Casio"
        if c_lower in {"levis", "levi's"}:
            return "Levi's"
        if c_lower in {"hm", "h&m"}:
            return "H&M"
        if c_lower in {"provogue men", "provogue"}:
            return "Provogue"
        if c_lower in {"roadster men", "roadster"}:
            return "Roadster"
        if c_lower in {"police men", "police"}:
            return "Police"
        if c_lower in {"arrow sport", "arrow woman", "arrow"}:
            return "Arrow"
        if c_lower in {"basics men", "basics"}:
            return "Basics"
        if c_lower in {"w women", "w"}:
            return "W"
        if c_lower in {"fossil"}:
            return "Fossil"
        if c_lower in {"timex"}:
            return "Timex"
        if c_lower in {"titan"}:
            return "Titan"
        if c_lower in {"fastrack"}:
            return "Fastrack"
        if c_lower in {"wrangler"}:
            return "Wrangler"
        if c_lower in {"nike"}:
            return "Nike"
        if c_lower in {"puma"}:
            return "Puma"
        if c_lower in {"fila"}:
            return "Fila"
        if c_lower in {"reebok"}:
            return "Reebok"
        if c_lower in {"converse"}:
            return "Converse"
        if c_lower in {"wildcraft"}:
            return "Wildcraft"
        if c_lower in {"catwalk"}:
            return "Catwalk"
        if c_lower in {"baggit"}:
            return "Baggit"
        if c_lower in {"fabindia"}:
            return "Fabindia"
        if c_lower in {"french connection"}:
            return "French Connection"
        if c_lower in {"flying machine"}:
            return "Flying Machine"
        if c_lower in {"peter england"}:
            return "Peter England"
        if c_lower in {"lino perros"}:
            return "Lino Perros"
        if c_lower in {"jealous 21"}:
            return "Jealous 21"
        return raw_candidate

    raw_candidates = []
    for n in names:
        tokens = n.strip().split()
        if not tokens:
            continue
        
        t0 = tokens[0]
        t0_lower = t0.lower()
        
        if len(tokens) >= 2:
            t1 = tokens[1]
            t1_lower = t1.lower()
            two_words = f"{t0} {t1}"
            two_words_lower = two_words.lower()

            if two_words_lower in KNOWN_2WORD_BRANDS:
                raw_candidates.append(canonical_brand(two_words))
                continue
            elif t0_lower in MAJOR_SINGLE_BRANDS:
                raw_candidates.append(canonical_brand(t0))
                continue
            elif t1_lower not in NON_BRAND_TOKEN2 and not re.match(r"^(ss\d+|\d+|men.*|women.*|boy.*|girl.*)", t1_lower):
                raw_candidates.append(canonical_brand(two_words))
                continue

        raw_candidates.append(canonical_brand(t0))

    brand_counts = Counter(raw_candidates)
    
    # Filter >= 20
    brands_ge_20 = {b: c for b, c in brand_counts.items() if c >= 20}
    # Sort descending
    brands_ge_20_sorted = dict(sorted(brands_ge_20.items(), key=lambda x: -x[1]))
    
    out_json = Path("src/brands.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(brands_ge_20_sorted, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(brands_ge_20_sorted)} brands (>= 20 items) to {out_json}")

    return brands_ge_20_sorted

if __name__ == "__main__":
    b_dict = extract_brand_candidates()
    print("\n--- TOP 100 BRAND CANDIDATES (>= 20 PRODUCTS) ---")
    for idx, (b, c) in enumerate(list(b_dict.items())[:100], 1):
        print(f"{idx:3d}. {b:<30}: {c:5d} items")
