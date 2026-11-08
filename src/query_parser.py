"""
Query Parser for Fashion Retrieval (src/query_parser.py)
Extracts structured attributes: articleType, gender, baseColour from natural Vietnamese queries.
Supports both accented and unaccented input.
"""
import re
from typing import Dict, List, Optional, Any

from src.vi_vocab import (
    remove_diacritics,
    GENDER_MAP,
    COLOUR_MAP,
    ARTICLE_TYPE_MAP
)

# Canonical keyword dictionary mapped to English schema
# Evaluated on unaccented lowercase tokens

GENDER_PATTERNS = [
    (r"\b(be trai|con trai|be nam)\b", "Boys"),
    (r"\b(be gai|con gai|be nu)\b", "Girls"),
    (r"\b(nam nu|unisex|cho ca nam va nu|dung chung)\b", "Unisex"),
    (r"\b(nam|cho nam|dan ong|phaimanh|men)\b", "Men"),
    (r"\b(nu|cho nu|phu nu|chi em|phaidep|women)\b", "Women"),
]

# Colour keywords
COLOUR_PATTERNS = [
    (r"\b(xanh navy|navy blue|xanh than|navy)\b", ["Navy Blue"]),
    (r"\b(xanh duong|blue|xanh lam|da troi)\b", ["Blue"]),
    (r"\b(xanh la|green|xanh luc|xanh bo)\b", ["Green"]),
    (r"\b(xanh ngoc|xanh mong ket|teal|turquoise)\b", ["Teal", "Turquoise Blue"]),
    (r"\b(xanh reu|o liu|olive)\b", ["Olive"]),
    (r"\b(den|black|huyen)\b", ["Black"]),
    (r"\b(trang nga|off white)\b", ["Off White"]),
    (r"\b(trang|white|bach)\b", ["White"]),
    (r"\b(xam melange|xam tieu|grey melange)\b", ["Grey Melange"]),
    (r"\b(xam than|charcoal)\b", ["Charcoal"]),
    (r"\b(xam|ghi|grey|gray)\b", ["Grey"]),
    (r"\b(do ruou|burgundy)\b", ["Burgundy"]),
    (r"\b(do do|do hat de|maroon)\b", ["Maroon"]),
    (r"\b(do|red)\b", ["Red"]),
    (r"\b(hong canh sen|magenta)\b", ["Magenta"]),
    (r"\b(hong dao|peach)\b", ["Peach"]),
    (r"\b(hong|pink)\b", ["Pink"]),
    (r"\b(vang kim|gold)\b", ["Gold"]),
    (r"\b(vang mu tat|mustard)\b", ["Mustard"]),
    (r"\b(vang|yellow)\b", ["Yellow"]),
    (r"\b(tim oai huong|lavender)\b", ["Lavender"]),
    (r"\b(tim|purple)\b", ["Purple"]),
    (r"\b(nau ca phe|coffee brown)\b", ["Coffee Brown"]),
    (r"\b(nau|brown)\b", ["Brown"]),
    (r"\b(be|beige)\b", ["Beige"]),
    (r"\b(kem|cream)\b", ["Cream"]),
    (r"\b(kaki|khaki)\b", ["Khaki"]),
    (r"\b(bac|silver)\b", ["Silver"]),
    (r"\b(dong|bronze|copper)\b", ["Bronze", "Copper"]),
    (r"\b(cam|orange)\b", ["Orange"]),
    (r"\b(nhieu mau|multi)\b", ["Multi"]),
]

# ArticleType keywords
ARTICLE_PATTERNS = [
    # Áo
    (r"\b(ao polo|polo)\b", ["Tshirts"]),
    (r"\b(ao thun|ao phong|tshirt|t-shirt|tee)\b", ["Tshirts"]),
    (r"\b(ao so mi|so mi|somi|shirt)\b", ["Shirts"]),
    (r"\b(ao khoac gio|ao gio|ao mua|rain jacket)\b", ["Rain Jacket", "Jackets"]),
    (r"\b(ao blazer|blazer)\b", ["Blazers", "Jackets"]),
    (r"\b(ao khoac|jacket)\b", ["Jackets"]),
    (r"\b(ao len|sweater)\b", ["Sweaters"]),
    (r"\b(ao hoodie|ao ni|sweatshirt)\b", ["Sweatshirts"]),
    (r"\b(ao sat nach|ao ba lo|ao may o|innerwear)\b", ["Innerwear Vests", "Tshirts"]),
    (r"\b(ao kieu|blouse|top|croptop)\b", ["Tops"]),
    (r"\b(ao kurta|kurta|kurti)\b", ["Kurtas", "Kurtis", "Kurta Sets"]),
    (r"\b(ao saree|saree|sari)\b", ["Sarees"]),
    # Quần & Váy
    (r"\b(quan jean|quan bo|jeans?)\b", ["Jeans"]),
    (r"\b(quan tay|quan au|quan cong so|trouser)\b", ["Trousers"]),
    (r"\b(quan kaki|kaki)\b", ["Trousers"]),
    (r"\b(quan short|quan dung|quan dui|short)\b", ["Shorts"]),
    (r"\b(quan jogger|quan the thao|track pant)\b", ["Track Pants"]),
    (r"\b(quan legging|quan bo om|legging|tight)\b", ["Leggings", "Tights"]),
    (r"\b(quan lot boxer|boxer|trunk)\b", ["Boxers", "Trunk"]),
    (r"\b(quan lot|brief)\b", ["Briefs"]),
    (r"\b(chan vay chu a|chan vay xoe|chan vay|skirt)\b", ["Skirts"]),
    (r"\b(vay dam|dam du tiec|dam suong|vay lien|dam|dress)\b", ["Dresses"]),
    (r"\b(jumpsuit|do bay)\b", ["Jumpsuit"]),
    (r"\b(do ngu|night suit|nightdress)\b", ["Night suits", "Nightdress"]),
    # Giày dép
    (r"\b(giay the thao|giay chay bo|giay da bong|giay tap gym|giay sneaker|sneaker)\b", ["Sports Shoes", "Casual Shoes"]),
    (r"\b(giay tay|giay da bong|giay cong so|formal shoe)\b", ["Formal Shoes"]),
    (r"\b(giay luoi|giay vai|giay canvas|giay casual|casual shoe)\b", ["Casual Shoes"]),
    (r"\b(giay cao got|cao got|heel)\b", ["Heels"]),
    (r"\b(giay bup be|giay bet|de bet|flat)\b", ["Flats"]),
    (r"\b(giay bot|bot da|bootie|boot)\b", ["Booties"]),
    (r"\b(sandal the thao|sports sandal)\b", ["Sports Sandals", "Sandals"]),
    (r"\b(dep quai hau|sandal|xang dan)\b", ["Sandals"]),
    (r"\b(dep kep|dep xo ngon|dep tong|flip flop)\b", ["Flip Flops"]),
    (r"\b(tat vo|tat|vo|sock)\b", ["Socks"]),
    # Túi balo ví
    (r"\b(balo laptop|laptop bag)\b", ["Laptop Bag", "Backpacks"]),
    (r"\b(balo du lich|balo phuot|balo|rucksack)\b", ["Backpacks", "Rucksacks"]),
    (r"\b(tui xach tay|tui xach nu|tui cong so|handbag)\b", ["Handbags"]),
    (r"\b(tui deo cheo|tui messenger|messenger bag)\b", ["Messenger Bag", "Handbags"]),
    (r"\b(tui bao tu|tui deo hong|waist pouch)\b", ["Waist Pouch"]),
    (r"\b(tui trong|duffel bag)\b", ["Duffel Bag"]),
    (r"\b(vi cam tay|clutch)\b", ["Clutches", "Wallets"]),
    (r"\b(vi da nam|vi da|bop tien|bop da|bop|vi|wallet)\b", ["Wallets"]),
    # Đồng hồ & Phụ kiện
    (r"\b(dong ho deo tay|dong ho|watch)\b", ["Watches"]),
    (r"\b(kinh ram|kinh mat|mat kinh|kinh phi cong|sunglass)\b", ["Sunglasses"]),
    (r"\b(that lung|day nit|day lung|belt)\b", ["Belts"]),
    (r"\b(ca vat|cavat|tie)\b", ["Ties"]),
    (r"\b(mu luoi trai|non ket|mu|non|cap)\b", ["Caps", "Hat"]),
    (r"\b(khan quang|khan len|scarf|muffler)\b", ["Scarves", "Mufflers"]),
    (r"\b(bong tai|hoa tai|khuyen tai|earring)\b", ["Earrings"]),
    (r"\b(day chuyen|vong co|mat trai tim|necklace|pendant)\b", ["Necklace and Chains", "Pendant"]),
    (r"\b(vong tay|lac tay|bangle|bracelet)\b", ["Bracelet", "Bangle"]),
    (r"\b(nhan|ring)\b", ["Ring"]),
]


def parse_query(query: str) -> Dict[str, Any]:
    """
    Parses a search query and extracts structured attributes:
    - articleType: list of possible articleType strings or None
    - gender: string (Men/Women/Boys/Girls/Unisex) or None
    - colour: list of baseColour strings or None
    """
    clean_q = remove_diacritics(query.lower()).strip()

    # 1. Extract Gender
    parsed_gender = None
    for pattern, gen in GENDER_PATTERNS:
        if re.search(pattern, clean_q):
            parsed_gender = gen
            break

    # 2. Extract Colour
    parsed_colour = None
    for pattern, cols in COLOUR_PATTERNS:
        if re.search(pattern, clean_q):
            parsed_colour = cols
            break

    # 3. Extract ArticleType
    parsed_article_types = None
    for pattern, arts in ARTICLE_PATTERNS:
        if re.search(pattern, clean_q):
            parsed_article_types = arts
            break

    return {
        "articleType": parsed_article_types,
        "gender": parsed_gender,
        "colour": parsed_colour,
        "raw_query": query
    }
