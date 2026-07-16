"""
Catalog of merchants per category, including realistic naming inconsistencies
(the kind you actually see on a bank/card statement): store numbers, POS
prefixes, abbreviations, trailing city/state codes, inconsistent casing.

Each category maps to a list of "merchant profiles". Each profile has a
canonical name plus a list of variant strings that will appear in the raw
transaction description. The generator picks a random variant so the same
real-world merchant shows up under several different strings, mirroring the
kind of messy text a categorization model has to learn to handle.
"""

MERCHANT_CATALOG = {
    "Food": [
        {
            "canonical": "Starbucks",
            "variants": [
                "STARBUCKS #04521", "SQ *STARBUCKS", "Starbucks Coffee",
                "STARBUCKS COFFEE CO", "SBUX 04521 SEATTLE WA",
            ],
            "amount_range": (3.5, 12.0),
        },
        {
            "canonical": "Whole Foods",
            "variants": [
                "WHOLEFDS MKT #102", "WHOLE FOODS MARKET", "WFM 102 AUSTIN TX",
                "Whole Foods",
            ],
            "amount_range": (15.0, 140.0),
        },
        {
            "canonical": "Chipotle",
            "variants": ["CHIPOTLE 2341", "CHIPOTLE MEXICAN GRILL", "Chipotle Online"],
            "amount_range": (8.0, 22.0),
        },
        {
            "canonical": "DoorDash",
            "variants": ["DOORDASH*RESTAURANT", "DD *DOORDASH", "DoorDash, Inc."],
            "amount_range": (12.0, 55.0),
        },
        {
            "canonical": "Trader Joe's",
            "variants": ["TRADER JOE S #543", "TRADER JOES", "Trader Joe's"],
            "amount_range": (10.0, 95.0),
        },
        {
            "canonical": "Local Diner",
            "variants": ["MAIN ST DINER", "THE CORNER DINER LLC", "Main Street Diner"],
            "amount_range": (9.0, 35.0),
        },
    ],
    "Shopping": [
        {
            "canonical": "Amazon",
            "variants": [
                "AMAZON.COM*A1B2C3", "AMZN Mktp US", "AMAZON MKTPLACE PMTS",
                "Amazon.com",
            ],
            "amount_range": (8.0, 220.0),
        },
        {
            "canonical": "Target",
            "variants": ["TARGET 00021453", "Target.com", "TARGET T-1453"],
            "amount_range": (12.0, 180.0),
        },
        {
            "canonical": "Best Buy",
            "variants": ["BEST BUY 00003210", "BESTBUY.COM", "Best Buy"],
            "amount_range": (25.0, 900.0),
        },
        {
            "canonical": "Nike",
            "variants": ["NIKE.COM", "NIKE STORE #221", "Nike Inc"],
            "amount_range": (30.0, 220.0),
        },
        {
            "canonical": "Etsy",
            "variants": ["ETSY.COM - SHOP12", "PAYPAL *ETSY", "Etsy Inc"],
            "amount_range": (10.0, 90.0),
        },
    ],
    "Travel": [
        {
            "canonical": "Delta Air Lines",
            "variants": ["DELTA AIR 0062134", "DELTA.COM", "DELTA AIRLINES"],
            "amount_range": (150.0, 850.0),
        },
        {
            "canonical": "Uber",
            "variants": ["UBER *TRIP", "UBER TECHNOLOGIES", "Uber Trip Help.uber.com"],
            "amount_range": (8.0, 60.0),
        },
        {
            "canonical": "Marriott",
            "variants": ["MARRIOTT HOTELS", "MARRIOTT 445621", "Marriott International"],
            "amount_range": (120.0, 650.0),
        },
        {
            "canonical": "Airbnb",
            "variants": ["AIRBNB HMXY23", "AIRBNB * HMXY23", "Airbnb Inc"],
            "amount_range": (90.0, 700.0),
        },
        {
            "canonical": "Enterprise Rent-A-Car",
            "variants": ["ENTERPRISE RENT-A-CAR", "ENTERPRISE RAC", "Enterprise"],
            "amount_range": (60.0, 400.0),
        },
    ],
    "Entertainment": [
        {
            "canonical": "Netflix",
            "variants": ["NETFLIX.COM", "NETFLIX 1888-638-3549"],
            "amount_range": (15.49, 22.99),
        },
        {
            "canonical": "Spotify",
            "variants": ["SPOTIFY USA", "SPOTIFY P1A2B3"],
            "amount_range": (10.99, 16.99),
        },
        {
            "canonical": "AMC Theatres",
            "variants": ["AMC 00234 ONLINE", "AMC THEATRES", "AMC.COM"],
            "amount_range": (12.0, 55.0),
        },
        {
            "canonical": "Steam",
            "variants": ["STEAMGAMES.COM", "VALVE CORP *STEAM"],
            "amount_range": (5.0, 70.0),
        },
    ],
    "Healthcare": [
        {
            "canonical": "CVS Pharmacy",
            "variants": ["CVS/PHARMACY #6532", "CVS PHARM", "CVS.COM"],
            "amount_range": (8.0, 90.0),
        },
        {
            "canonical": "Planet Fitness",
            "variants": ["PLANET FITNESS", "PF #0233 MONTHLY"],
            "amount_range": (10.0, 25.0),
        },
        {
            "canonical": "City Medical Group",
            "variants": ["CITY MEDICAL GRP", "CITY MED GROUP CO-PAY"],
            "amount_range": (20.0, 250.0),
        },
    ],
    "Utilities": [
        {
            "canonical": "Pacific Gas & Electric",
            "variants": ["PGE WEB PAYMENT", "PACIFIC GAS ELEC", "PG&E"],
            "amount_range": (60.0, 220.0),
        },
        {
            "canonical": "Comcast Xfinity",
            "variants": ["COMCAST XFINITY", "XFINITY-ONLINE PMT"],
            "amount_range": (65.0, 130.0),
        },
        {
            "canonical": "Verizon Wireless",
            "variants": ["VERIZON WRLS", "VZWRLSS*BILL PAY"],
            "amount_range": (45.0, 140.0),
        },
    ],
    "Education": [
        {
            "canonical": "Coursera",
            "variants": ["COURSERA INC", "COURSERA.ORG"],
            "amount_range": (39.0, 79.0),
        },
        {
            "canonical": "State University",
            "variants": ["STATE UNIV TUITION", "STATE UNIVERSITY BURSAR"],
            "amount_range": (500.0, 4500.0),
        },
    ],
    "Bills": [
        {
            "canonical": "Rent Payment",
            "variants": ["RENT - PROPERTY MGMT CO", "ACH RENT PAYMENT"],
            "amount_range": (1200.0, 2600.0),
        },
        {
            "canonical": "State Farm Insurance",
            "variants": ["STATE FARM INS", "STATEFARM AUTOPAY"],
            "amount_range": (80.0, 220.0),
        },
    ],
    "Investment": [
        {
            "canonical": "Fidelity",
            "variants": ["FIDELITY INV", "FIDELITY BROKERAGE"],
            "amount_range": (100.0, 2000.0),
        },
        {
            "canonical": "Vanguard",
            "variants": ["VANGUARD GRP", "VANGUARD.COM"],
            "amount_range": (100.0, 2000.0),
        },
    ],
    "Miscellaneous": [
        {
            "canonical": "ATM Withdrawal",
            "variants": ["ATM WITHDRAWAL #4521", "CASH WITHDRAWAL ATM"],
            "amount_range": (20.0, 200.0),
        },
        {
            "canonical": "Venmo Transfer",
            "variants": ["VENMO PAYMENT", "VENMO*TRANSFER"],
            "amount_range": (5.0, 150.0),
        },
    ],
}

# Recurring monthly subscriptions (fixed-ish amount, same day each month +/- jitter)
RECURRING_SUBSCRIPTIONS = [
    {"canonical": "Netflix", "variant": "NETFLIX.COM", "category": "Entertainment", "amount": 15.49, "day": 3},
    {"canonical": "Spotify", "variant": "SPOTIFY USA", "category": "Entertainment", "amount": 10.99, "day": 7},
    {"canonical": "Planet Fitness", "variant": "PF #0233 MONTHLY", "category": "Healthcare", "amount": 24.99, "day": 1},
    {"canonical": "Rent Payment", "variant": "RENT - PROPERTY MGMT CO", "category": "Bills", "amount": 1850.00, "day": 1},
    {"canonical": "Comcast Xfinity", "variant": "COMCAST XFINITY", "category": "Utilities", "amount": 89.99, "day": 12},
    {"canonical": "Verizon Wireless", "variant": "VERIZON WRLS", "category": "Utilities", "amount": 75.00, "day": 15},
    {"canonical": "State Farm Insurance", "variant": "STATE FARM INS", "category": "Bills", "amount": 142.00, "day": 20},
]

# Seasonal multipliers by month (1-12) applied to category selection weight
SEASONAL_MULTIPLIERS = {
    "Shopping": {11: 1.8, 12: 2.2, 1: 1.3},          # holiday shopping + Jan returns/sales
    "Travel":   {6: 1.6, 7: 1.9, 8: 1.6, 12: 1.4},   # summer + winter holidays
    "Food":     {11: 1.2, 12: 1.3},                   # holiday gatherings
}
