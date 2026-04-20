"""Structured waste knowledge base for Smart Waste Vision.

Each waste category has disposal instructions, environmental impact data,
educational information, and safety alerts used across the app.
"""

WASTE_KNOWLEDGE = {
    "battery": {
        "disposal": {
            "status": "hazardous",
            "bin_color": "red",
            "bin_label": "Hazardous Waste",
            "steps": [
                "Place tape over battery terminals to prevent short circuits",
                "Store in a cool, dry place until disposal",
                "Take to a battery recycling drop-off or hazardous waste facility",
                "Never place in regular trash or recycling bins",
            ],
            "common_mistakes": [
                "Throwing batteries in regular trash (fire risk in waste trucks)",
                "Mixing different battery types together",
                "Putting lithium batteries in curbside recycling",
            ],
        },
        "environmental_impact": {
            "decomposition_time": "100+ years",
            "recycling_benefit": "Recovers lithium, cobalt, and nickel — reduces mining demand by up to 70%",
            "fun_fact": "A single battery can contaminate 600,000 liters of water if it leaks in a landfill",
        },
        "education": {
            "example_items": [
                "AA/AAA batteries", "button cells", "9V batteries",
                "laptop/phone batteries", "rechargeable packs", "car batteries",
            ],
            "description": "Portable energy storage devices containing heavy metals and toxic chemicals that require special disposal.",
        },
        "safety": {
            "level": "danger",
            "alert": "Contains toxic chemicals (lead, cadmium, lithium). Never incinerate — risk of explosion. Keep away from children and water.",
        },
    },
    "biological": {
        "disposal": {
            "status": "compostable",
            "bin_color": "green",
            "bin_label": "Compost / Organic Waste",
            "steps": [
                "Separate from non-organic waste (remove packaging, stickers, ties)",
                "Place in a compost bin or green waste container",
                "If home composting, mix greens (food scraps) with browns (leaves, cardboard)",
                "Seal the bin to prevent pests and odors",
            ],
            "common_mistakes": [
                "Including meat or dairy in home compost (attracts pests — OK for municipal composting)",
                "Leaving food waste in plastic bags (use compostable bags or go bagless)",
                "Mixing organic waste with recyclables (contaminates the recycling stream)",
            ],
        },
        "environmental_impact": {
            "decomposition_time": "1 week to 6 months",
            "recycling_benefit": "Composting diverts waste from landfills and produces nutrient-rich soil amendment",
            "fun_fact": "Food waste in landfills generates methane, a greenhouse gas 80x more potent than CO2 over 20 years",
        },
        "education": {
            "example_items": [
                "Fruit and vegetable scraps", "coffee grounds and filters", "eggshells",
                "yard trimmings", "leaves and grass clippings", "tea bags (non-plastic)",
            ],
            "description": "Organic matter from food, plants, and garden waste that can decompose naturally and return nutrients to the soil.",
        },
        "safety": {
            "level": "warning",
            "alert": "May attract pests and produce odors — dispose promptly. Wash hands after handling. Avoid contact with mold or decaying matter.",
        },
    },
    "cardboard": {
        "disposal": {
            "status": "recyclable",
            "bin_color": "blue",
            "bin_label": "Recycling",
            "steps": [
                "Remove all tape, staples, and shipping labels",
                "Flatten boxes to save space in the recycling bin",
                "Keep dry — wet or greasy cardboard goes in compost/trash",
                "Place in the blue recycling bin",
            ],
            "common_mistakes": [
                "Recycling pizza boxes with grease stains (compost them instead)",
                "Leaving packing peanuts or bubble wrap inside boxes",
                "Not flattening boxes (wastes space and causes collection problems)",
            ],
        },
        "environmental_impact": {
            "decomposition_time": "2 to 3 months",
            "recycling_benefit": "Recycling 1 ton of cardboard saves 9 cubic yards of landfill space and 46 gallons of oil",
            "fun_fact": "Cardboard can be recycled 5-7 times before the fibers become too short",
        },
        "education": {
            "example_items": [
                "Shipping boxes", "cereal boxes", "shoe boxes",
                "toilet paper rolls", "egg cartons", "moving boxes",
            ],
            "description": "Paper-based packaging material made from wood pulp. One of the most commonly recycled materials worldwide.",
        },
        "safety": {
            "level": "info",
            "alert": "Generally safe to handle. Watch for staples or sharp edges on corrugated cardboard.",
        },
    },
    "clothes": {
        "disposal": {
            "status": "recyclable",
            "bin_color": "blue",
            "bin_label": "Textile Recycling / Donation",
            "steps": [
                "Wash and dry the clothing if possible",
                "If wearable, donate to charity shops or clothing banks",
                "If unwearable, place in a textile recycling drop-off bin",
                "Cut unusable fabrics into cleaning rags before discarding",
            ],
            "common_mistakes": [
                "Throwing clothes in regular trash (textiles can take 200+ years to decompose)",
                "Donating wet or mold-contaminated clothing",
                "Mixing textiles with paper/cardboard recycling (different process)",
            ],
        },
        "environmental_impact": {
            "decomposition_time": "20 to 200+ years (synthetic fabrics take the longest)",
            "recycling_benefit": "Extending a garment's life by 9 months reduces its carbon footprint by 20-30%",
            "fun_fact": "The fashion industry produces 10% of global carbon emissions — more than international flights and shipping combined",
        },
        "education": {
            "example_items": [
                "T-shirts and shirts", "jeans and pants", "jackets and coats",
                "socks and underwear", "scarves and hats", "towels and bed linens",
            ],
            "description": "Wearable textiles made from natural (cotton, wool) or synthetic (polyester, nylon) fibers.",
        },
        "safety": {
            "level": "info",
            "alert": "Generally safe. Check pockets for sharp objects or batteries before recycling. Wash to remove allergens.",
        },
    },
    "glass": {
        "disposal": {
            "status": "recyclable",
            "bin_color": "blue",
            "bin_label": "Glass Recycling",
            "steps": [
                "Rinse the container to remove food residue",
                "Remove metal lids and caps (recycle those with metals)",
                "Do NOT break the glass — keep containers whole",
                "Place in glass recycling bin (separate by color if required locally)",
            ],
            "common_mistakes": [
                "Mixing window glass or mirrors with container glass (different melting points)",
                "Including ceramic, porcelain, or Pyrex (contaminates the batch)",
                "Leaving food residue inside (attracts pests and reduces recyclability)",
            ],
        },
        "environmental_impact": {
            "decomposition_time": "1 million+ years",
            "recycling_benefit": "Recycling glass reduces energy use by 30% vs. making new glass from raw materials",
            "fun_fact": "Glass is 100% recyclable and can be recycled endlessly without losing quality or purity",
        },
        "education": {
            "example_items": [
                "Bottles (wine, beer, soda)", "jars (jam, sauce, pickles)",
                "Cosmetic containers", "food storage jars",
            ],
            "description": "Containers made from silica sand, soda ash, and limestone. One of the most sustainably recyclable materials.",
        },
        "safety": {
            "level": "warning",
            "alert": "Handle with care — broken glass causes cuts. Wrap broken pieces in newspaper or thick paper before disposal. Do not place broken glass in recycling.",
        },
    },
    "metal": {
        "disposal": {
            "status": "recyclable",
            "bin_color": "blue",
            "bin_label": "Metal Recycling",
            "steps": [
                "Rinse cans and tins to remove food residue",
                "Remove paper labels if they come off easily (optional)",
                "Crush cans to save space (optional)",
                "Place in the metal/recycling bin — aluminum and steel accepted",
            ],
            "common_mistakes": [
                "Recycling aerosol cans that are not fully empty (pressure hazard)",
                "Including metal items with electronic components (e-waste, not metal recycling)",
                "Throwing sharp metal lids loose in the bin (nest them inside the can)",
            ],
        },
        "environmental_impact": {
            "decomposition_time": "50 to 500 years",
            "recycling_benefit": "Recycling aluminum saves 95% of the energy needed to produce it from raw ore",
            "fun_fact": "Aluminum can be recycled infinitely — a can returns to the shelf as a new can in just 60 days",
        },
        "education": {
            "example_items": [
                "Soda and beer cans", "food tins (soup, beans)", "aluminum foil (clean)",
                "metal bottle caps", "empty aerosol cans", "steel food containers",
            ],
            "description": "Aluminum and steel containers widely used for food and beverage packaging. Among the most valuable recyclable materials.",
        },
        "safety": {
            "level": "warning",
            "alert": "Sharp edges on opened cans and lids — handle carefully. Wear gloves if handling rusty metal. Ensure aerosol cans are fully depressurized.",
        },
    },
    "paper": {
        "disposal": {
            "status": "recyclable",
            "bin_color": "blue",
            "bin_label": "Paper Recycling",
            "steps": [
                "Keep paper clean and dry",
                "Remove any plastic windows from envelopes",
                "Shredded paper should be placed in a paper bag (loose shreds jam machines)",
                "Place in the paper recycling bin",
            ],
            "common_mistakes": [
                "Recycling paper contaminated with food or grease (compost it instead)",
                "Including receipts (thermal paper contains BPA and cannot be recycled)",
                "Recycling wax-coated or laminated paper (goes to trash)",
            ],
        },
        "environmental_impact": {
            "decomposition_time": "2 to 6 weeks",
            "recycling_benefit": "Recycling 1 ton of paper saves 17 trees, 7,000 gallons of water, and 3 cubic yards of landfill space",
            "fun_fact": "Paper can be recycled 5-7 times before fibers become too short, then it can be composted",
        },
        "education": {
            "example_items": [
                "Newspapers and magazines", "office paper and envelopes",
                "junk mail and catalogs", "paper bags", "wrapping paper (non-metallic)",
                "notebooks and school papers",
            ],
            "description": "Thin sheets made from wood pulp fibers. Paper recycling is one of the oldest and most widespread recycling processes.",
        },
        "safety": {
            "level": "info",
            "alert": "Generally safe. Paper cuts are a minor risk. Avoid recycling paper with unknown chemical contamination.",
        },
    },
    "plastic": {
        "disposal": {
            "status": "recyclable",
            "bin_color": "blue",
            "bin_label": "Plastic Recycling",
            "steps": [
                "Check the resin identification code (triangle number on the bottom)",
                "Rinse containers to remove food residue",
                "Replace caps on bottles (caps are recycled separately at the facility)",
                "Codes 1 (PET) and 2 (HDPE) are widely accepted — check local rules for 3-7",
            ],
            "common_mistakes": [
                "Recycling plastic bags in curbside bins (they jam sorting machines — use store drop-offs)",
                "Including styrofoam/polystyrene (code 6 — rarely accepted curbside)",
                "Not rinsing containers (food residue contaminates entire batches)",
                "Wishful recycling — putting non-recyclable plastics in the bin hoping they will get recycled",
            ],
        },
        "environmental_impact": {
            "decomposition_time": "20 to 500+ years (varies by plastic type)",
            "recycling_benefit": "Recycling 1 ton of plastic saves 5,774 kWh of energy and 16.3 barrels of oil",
            "fun_fact": "Only about 9% of all plastic ever produced has been recycled — the rest is in landfills, incinerated, or in the environment",
        },
        "education": {
            "example_items": [
                "Water/soda bottles (PET #1)", "milk jugs and detergent bottles (HDPE #2)",
                "Yogurt containers (PP #5)", "plastic bags and wrap (LDPE #4)",
                "Styrofoam containers (PS #6)", "food packaging and clamshells",
            ],
            "description": "Synthetic polymer materials derived from petroleum. Different resin codes have vastly different recyclability.",
        },
        "safety": {
            "level": "info",
            "alert": "Avoid burning plastic — releases toxic fumes including dioxins. Some plastics (PVC #3) contain harmful chemicals. Wash hands after handling dirty containers.",
        },
    },
    "shoes": {
        "disposal": {
            "status": "recyclable",
            "bin_color": "blue",
            "bin_label": "Textile / Shoe Recycling",
            "steps": [
                "Clean shoes and tie pairs together with laces",
                "If wearable, donate to charity shops or shoe donation programs",
                "If unwearable, use a shoe/textile recycling drop-off bin",
                "Some athletic brands (Nike, Adidas) accept old shoes for recycling programs",
            ],
            "common_mistakes": [
                "Throwing shoes in regular trash (mixed materials take centuries to decompose)",
                "Donating shoes with hazardous damage (mold, chemical spills)",
                "Separating a pair — always donate/recycle both shoes together",
            ],
        },
        "environmental_impact": {
            "decomposition_time": "25 to 40+ years",
            "recycling_benefit": "Recycled shoes are ground into rubber crumb for playground surfaces, athletic tracks, and insulation",
            "fun_fact": "About 300 million pairs of shoes end up in landfills each year in the US alone",
        },
        "education": {
            "example_items": [
                "Sneakers and athletic shoes", "boots and sandals",
                "dress shoes and heels", "flip-flops and slippers",
            ],
            "description": "Footwear made from mixed materials (rubber, leather, textiles, plastics). Challenging to recycle due to material diversity.",
        },
        "safety": {
            "level": "info",
            "alert": "Generally safe. Remove insoles before recycling. Check for embedded nails or sharp objects in old boots.",
        },
    },
    "trash": {
        "disposal": {
            "status": "landfill",
            "bin_color": "black",
            "bin_label": "General Waste (Landfill)",
            "steps": [
                "Confirm the item truly cannot be recycled, composted, or donated",
                "Check if any component can be separated (e.g., remove batteries from electronics)",
                "Bag securely to prevent litter and contain odors",
                "Place in the black/general waste bin",
            ],
            "common_mistakes": [
                "Sending recyclable items to landfill without checking (when in doubt, look it up)",
                "Mixing hazardous waste with general trash (batteries, chemicals, paint)",
                "Not separating multi-material items (e.g., a plastic toy with batteries inside)",
            ],
        },
        "environmental_impact": {
            "decomposition_time": "Varies widely (some items never fully decompose)",
            "recycling_benefit": "Reducing landfill waste by just 10% can save millions of tons of CO2 equivalent annually",
            "fun_fact": "The average person generates about 4.4 pounds (2 kg) of waste per day — reducing this is the highest-impact action",
        },
        "education": {
            "example_items": [
                "Chip bags and candy wrappers", "broken ceramics and porcelain",
                "Used paper towels and tissues", "diapers and hygiene products",
                "Heavily contaminated food packaging", "broken toys (non-electronic)",
            ],
            "description": "Non-recyclable, non-compostable waste that must go to landfill. The goal is to minimize this category through better sorting.",
        },
        "safety": {
            "level": "warning",
            "alert": "May contain mixed hazards — use gloves when handling. Look for hidden recyclables or hazardous items before discarding.",
        },
    },
}

# Display configuration for disposal status
BIN_DISPLAY = {
    "recyclable":  {"emoji": "♻️",  "color": "#2196F3", "bin": "Blue Bin"},
    "compostable": {"emoji": "🌱", "color": "#4CAF50", "bin": "Green Bin"},
    "hazardous":   {"emoji": "⚠️",  "color": "#F44336", "bin": "Red Bin / Special Drop-off"},
    "landfill":    {"emoji": "🗑️",  "color": "#616161", "bin": "Black Bin"},
}

# Display configuration for safety levels
SAFETY_DISPLAY = {
    "danger":  {"icon": "🔴", "label": "DANGER"},
    "warning": {"icon": "🟡", "label": "WARNING"},
    "info":    {"icon": "🔵", "label": "INFO"},
}
