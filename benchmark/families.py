"""Families of look-alike documents, the hard part of real life.

Each family writes many similar files (24 monthly statements, 60 receipts from 20
stores, 40 job postings, 150 notes...) and the searches that should tell them apart,
phrased the way people ask rather than copied from the documents. Everything is drawn
from fixed lists with a seeded random generator, so the output is identical every time.
All people, companies, numbers and places are invented.
"""

from __future__ import annotations

import calendar
import random
from dataclasses import dataclass, field
from pathlib import Path

import writers as w
from writers import docx, ods, pdf, pptx, scanned_pdf, text_image, xlsx

MONTHS = list(calendar.month_name)[1:]


@dataclass
class Out:
    root: Path
    files: list[str] = field(default_factory=list)
    queries: list[dict] = field(default_factory=list)
    receipt_stores: dict[str, list[str]] = field(default_factory=dict)

    def path(self, rel: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        self.files.append(rel)
        return p

    def ask(self, q: str, cat: str, *want: str) -> None:
        self.queries.append({"q": q, "cat": cat, "want": list(want)})


# make_corpus.py writes some of these by hand, with details its own searches ask about.
# The families skip those, so no month or trip has two contradicting documents.
HAND_WRITTEN_MONTHS = {"statement": {(2024, 5)}, "electric": {(2024, 3)}}
HAND_WRITTEN_TRIPS = {
    "Lisbon": ("Travel/flight-confirmation-lisbon.pdf", "Travel/hotel-booking-lisbon.html")
}


# -- money ----------------------------------------------------------------------------------


def bank_statements(o: Out) -> None:
    rnd = random.Random(101)
    shops = ["Green Basket Market", "Hartley Hardware", "Fuel Stop #22", "Corner Bistro", "StreamFlix",
             "Riverside Pharmacy", "Bookworm Books", "City Parking", "Sunny Side Cafe", "PetPal Supplies"]  # fmt: skip
    special = {(2023, 2): "02/14  Overdraft fee  -35.00", (2024, 9): "09/12  Refund - Atlantic Air  +412.00",
               (2024, 4): "04/20  Wire to Lakeside Escrow (security deposit)  -1,850.00"}  # fmt: skip
    balance = 3100.0
    for year in (2023, 2024):
        for m in range(1, 13):
            if (year, m) in HAND_WRITTEN_MONTHS["statement"]:
                continue
            lines = [f"{m:02d}/01  Rent - Lakeside Property Mgmt  -1,850.00",
                     f"{m:02d}/03  Payroll deposit - Brightline Logistics  +2,615.40",
                     f"{m:02d}/17  Payroll deposit - Brightline Logistics  +2,615.40"]  # fmt: skip
            for _ in range(rnd.randint(5, 9)):
                lines.append(
                    f"{m:02d}/{rnd.randint(2, 28):02d}  {rnd.choice(shops)}  -{rnd.uniform(4, 180):.2f}"
                )
            if (year, m) in special:
                lines.append(special[(year, m)])
            balance += rnd.uniform(-300, 600)
            body = "\n".join(sorted(lines))
            pdf(o.path(f"Finance/Statements/cedar-checking-{year}-{m:02d}.pdf"), [
                f"CEDAR CREDIT UNION - CHECKING STATEMENT\nAccount ending 4471\nStatement period: "
                f"{MONTHS[m - 1]} 1 - {MONTHS[m - 1]} {calendar.monthrange(year, m)[1]}, {year}\n\n{body}\n\n"
                f"Closing balance: ${balance:,.2f}"])  # fmt: skip
    for year, m in ((2023, 5), (2024, 1), (2024, 10)):
        o.ask(f"bank statement for {MONTHS[m - 1]} {year}", "describe",
              f"Finance/Statements/cedar-checking-{year}-{m:02d}.pdf")  # fmt: skip
    o.ask(
        "the month I got charged an overdraft fee",
        "question",
        "Finance/Statements/cedar-checking-2023-02.pdf",
    )
    o.ask(
        "statement showing the airline refund",
        "question",
        "Finance/Statements/cedar-checking-2024-09.pdf",
    )


def utility_bills(o: Out) -> None:
    rnd = random.Random(102)
    for year in (2023, 2024):
        for m in range(1, 13):
            if (year, m) in HAND_WRITTEN_MONTHS["electric"]:
                continue
            kwh = int(420 + 260 * abs(6.5 - m) / 6 + rnd.randint(-40, 40))
            amount = kwh * 0.118 + 12.9
            rel = f"Bills/Electric/metro-power-{year}-{m:02d}.pdf"
            pdf(o.path(rel), [
                f"METRO POWER & LIGHT\nAccount 3301-772-845\nBilling month: {MONTHS[m - 1]} {year}\n"
                f"Energy used: {kwh} kWh\nAmount due: ${amount:.2f}\nPay online or by phone."])  # fmt: skip
    for m in range(1, 13):
        gallons = rnd.randint(2400, 4100) + (2600 if m == 7 else 0)
        note = "\nUsage is much higher than usual. Check for leaks." if m == 7 else ""
        pdf(o.path(f"Bills/Water/springfield-water-2024-{m:02d}.pdf"), [
            f"SPRINGFIELD WATER DEPARTMENT\nService address: 88 Willow Court\nPeriod: {MONTHS[m - 1]} 2024\n"
            f"Water used: {gallons} gallons\nAmount due: ${gallons * 0.011 + 18:.2f}{note}"])  # fmt: skip
        extra = ""
        if m == 8:
            extra = "\nInternational roaming (Portugal): $64.00\nRoaming data pass 2 GB: $35.00"
        rel = f"Bills/Phone/northstar-mobile-2024-{m:02d}.pdf"
        pdf(o.path(rel), [
            f"NORTHSTAR MOBILE\nAccount: Rivera family plan (3 lines)\nStatement: {MONTHS[m - 1]} 2024\n"
            f"Plan: Unlimited Talk & Text, 30 GB shared data  $95.00{extra}\nTaxes and fees: $11.38"])  # fmt: skip
    for year, m in ((2023, 1), (2023, 8), (2024, 12)):
        o.ask(
            f"electric bill for {MONTHS[m - 1]} {year}",
            "describe",
            f"Bills/Electric/metro-power-{year}-{m:02d}.pdf",
        )
    o.ask(
        "water bill from this past March", "describe", "Bills/Water/springfield-water-2024-03.pdf"
    )
    o.ask(
        "the water bill that warned about a leak",
        "question",
        "Bills/Water/springfield-water-2024-07.pdf",
    )
    o.ask(
        "phone bill with the roaming charges from Portugal",
        "question",
        "Bills/Phone/northstar-mobile-2024-08.pdf",
    )


def paystubs(o: Out) -> None:
    day = 12  # biweekly Fridays in 2024 starting January 12
    from datetime import date, timedelta

    first = date(2024, 1, day)
    rels = []
    for i in range(26):
        d = first + timedelta(days=14 * i)
        bonus = "\nAnnual performance bonus: $3,000.00" if i == 24 else ""
        overtime = "\nOvertime 9.5 hours: $431.25" if i == 7 else ""
        if d.isoformat() == "2024-06-14":
            rels.append("Finance/paystub-june-2024.pdf")  # hand-written, more detailed
            continue
        rel = f"Finance/Paystubs/brightline-paystub-{d.isoformat()}.pdf"
        rels.append(rel)
        pdf(o.path(rel), [
            f"BRIGHTLINE LOGISTICS - EARNINGS STATEMENT\nEmployee: Sam Rivera\nPay date: {d.strftime('%B %d, %Y')}\n"
            f"Regular pay: $3,153.85{overtime}{bonus}\nFederal tax withheld, Social Security, Medicare, 401(k)\n"
            "Direct deposit to Cedar Credit Union ****4471"])  # fmt: skip
    o.ask("pay stub that included my bonus", "question", rels[24])
    o.ask("the paycheck with overtime", "question", rels[7])
    o.ask("pay stub from the first paycheck of the year", "describe", rels[0])


def tax_forms(o: Out) -> None:
    pdf(o.path("Taxes/2021-tax-return-summary.pdf"), [
        "FEDERAL INCOME TAX RETURN SUMMARY - TAX YEAR 2021\nFiling status: Married filing jointly\n"
        "Refund amount: $640.00\nPrepared by Moreno & Associates, CPA"])  # fmt: skip
    for year, wages in ((2021, "61,200.00"), (2022, "74,900.00")):
        scanned_pdf(o.path(f"Taxes/w2-{year}-scan.pdf"), [[
            f"**Form W-2 Wage and Tax Statement {year}", "Employer: Brightline Logistics Inc.",
            "Employee: Sam Rivera", f"1 Wages, tips, other compensation: {wages}"]], seed=200 + year)  # fmt: skip
    for year, interest in ((2022, "38.12"), (2023, "211.47")):
        pdf(o.path(f"Taxes/1099-INT-{year}-cedar.pdf"), [
            f"FORM 1099-INT INTEREST INCOME {year}\nPayer: Cedar Credit Union\nRecipient: Sam Rivera\n"
            f"Box 1 Interest income: ${interest}"])  # fmt: skip
    o.ask("2021 tax return", "describe", "Taxes/2021-tax-return-summary.pdf")
    o.ask("W-2 from 2022", "visual", "Taxes/w2-2022-scan.pdf")
    o.ask(
        "interest income form from the credit union for 2023",
        "describe",
        "Taxes/1099-INT-2023-cedar.pdf",
    )


# -- receipts ---------------------------------------------------------------------------------

STORES = {
    "TECHZONE ELECTRONICS": ["USB-C charger", "HDMI cable 6ft", "wireless mouse", "printer ink XL", "noise cancelling earbuds"],
    "BLOOM & STEM FLORIST": ["dozen red roses", "white orchid in pot", "sunflower bouquet"],
    "PAGE TURNER BOOKS": ["The Midnight Library", "crossword puzzle book", "kids atlas"],
    "SUMMIT OUTDOOR CO.": ["2-person tent", "hiking boots", "headlamp", "water filter"],
    "TIRE & WHEEL PROS": ["4 snow tires", "wheel alignment", "tire rotation"],
    "HARMONY MUSIC SHOP": ["violin strings set", "guitar picks", "ukulele tuner"],
    "PAWS PET SUPPLY": ["dog food 30lb", "cat litter", "chew toy", "flea treatment"],
    "KITCHEN COMFORTS": ["air fryer", "cast iron skillet", "knife sharpener", "silicone spatulas"],
    "BRIGHT SMILE DENTAL": ["teeth cleaning", "x-rays", "fluoride treatment"],
    "QUICKLUBE SPRINGFIELD": ["synthetic oil change", "cabin air filter", "wiper blades"],
    "CINEMA 8": ["2 adult tickets", "large popcorn", "kids ticket"],
    "SWEET CRUMBS BAKERY": ["birthday cake custom", "croissants x6", "sourdough loaf"],
    "OFFICE DEPOT PLUS": ["desk chair", "printer paper case", "standing desk mat"],
    "GARDEN GROVE NURSERY": ["tomato seedlings", "potting soil 2 bags", "lemon tree"],
    "SPARKLE CAR WASH": ["premium wash", "interior detail"],
    "TOYLAND": ["LEGO castle set", "puzzle 500 pieces", "jump rope"],
    "VISION CENTER": ["eye exam", "prescription glasses", "contact lens solution"],
    "HOME COMFORT HVAC": ["furnace tune-up", "air filter 16x25"],
    "FRESHWAY GROCERY": ["salmon fillet", "avocados", "almond milk", "blueberries", "pasta"],
    "CORNER BISTRO": ["club sandwich", "tomato soup", "iced latte"],
}  # fmt: skip


def receipts(o: Out) -> None:
    rnd = random.Random(103)
    names = list(STORES)
    item_receipts: dict[str, list[str]] = {}
    per_store: dict[str, list[str]] = {}
    for i in range(60):
        store = names[i % len(names)]
        items = rnd.sample(STORES[store], k=min(len(STORES[store]), rnd.randint(1, 3)))
        prices = [round(rnd.uniform(4, 220), 2) for _ in items]
        lines = [
            f"**{store}",
            f"{rnd.randint(1, 12):02d}/{rnd.randint(1, 28):02d}/2024 {rnd.randint(8, 20)}:{rnd.randint(0, 59):02d}",
            "",
        ]
        lines += [f"{it[:22]:22} {p:8.2f}" for it, p in zip(items, prices, strict=True)]
        lines += [
            "",
            f"**TOTAL {sum(prices):15.2f}",
            rnd.choice(["VISA ****4471", "MASTERCARD ****9020", "CASH"]),
        ]
        img = w.photo_of(
            text_image(lines, width=640, mono=True, size=24, bg=(250, 249, 244)), 300 + i
        )
        rel = f"Pictures/Receipts/IMG_{2400 + i * 3}.jpg"
        img.save(o.path(rel), quality=86)
        per_store.setdefault(store, []).append(rel)
        for it in items:
            item_receipts.setdefault(it, []).append(rel)
    o.receipt_stores = per_store
    # Items bought only once: "the receipt for the X" then has exactly one right answer.
    unique = [(it, rels[0]) for it, rels in item_receipts.items() if len(rels) == 1]
    rnd.shuffle(unique)
    phrasing = [
        "receipt for the {}",
        "where's the receipt for the {}",
        "how much did I pay for the {}",
    ]
    for n, (item, rel) in enumerate(unique[:22]):
        o.ask(phrasing[n % 3].format(item.lower()), "visual" if n % 3 != 2 else "question", rel)
    for store, rels in per_store.items():
        if len(rels) == 3 and store in ("SUMMIT OUTDOOR CO.", "VISION CENTER", "TOYLAND"):
            o.ask(f"receipts from {store.title()}", "visual", *rels)


# -- work -------------------------------------------------------------------------------------

COMPANIES = ["Acme Analytics", "Northpeak Health", "Bluefin Robotics", "Juniper Bank", "Orbit Games",
             "Cobalt Energy", "Tidewater Shipping", "Lumen Labs", "Vesta Insurance", "Kestrel Air"]  # fmt: skip
ROLES = ["Data Analyst", "Senior Data Engineer", "Product Manager", "Analytics Manager"]
CITIES = [
    "Chicago",
    "Denver",
    "Austin",
    "Boston",
    "Seattle",
    "Atlanta",
    "Remote",
    "Portland",
    "Miami",
    "Raleigh",
]


def job_postings(o: Out) -> None:
    rnd = random.Random(104)
    fmts = ["txt", "html", "pdf", "docx"]
    asked = 0
    for i, (company, role) in enumerate((c, r) for c in COMPANIES for r in ROLES):
        city = CITIES[(i * 7) % len(CITIES)]
        low = 70 + rnd.randint(0, 9) * 5 + (40 if "Senior" in role or "Manager" in role else 0)
        perk = rnd.choice(["four-day work week", "fully paid parental leave", "annual learning budget",
                           "relocation package", "on-site gym", "dog-friendly office"])  # fmt: skip
        text = (f"{role} - {company} ({city})\n\nWe are hiring a {role.lower()} to join our team. You will "
                f"work with SQL, Python and dashboards. Salary: ${low},000 - ${low + 25},000. Benefits "
                f"include a {perk}.")  # fmt: skip
        ext = fmts[i % 4]
        rel = f"Career/Job postings/{company.replace(' ', '-')}-{role.replace(' ', '-')}.{ext}"
        if ext == "txt":
            o.path(rel).write_text(text)
        elif ext == "html":
            o.path(rel).write_text(f"<html><head><title>{role} at {company}</title></head><body><h1>{role}"
                                   f"</h1><p>{text}</p></body></html>")  # fmt: skip
        elif ext == "pdf":
            pdf(o.path(rel), [text])
        else:
            docx(o.path(rel), text.split("\n\n"))
        if i % 3 == 0 and asked < 12:
            o.ask(f"the {role.lower()} job at {company}", "describe", rel)
            asked += 1
        if perk == "four-day work week" and city != "Remote" and "Product" in role:
            o.ask(f"product manager job with a four-day week in {city}", "question", rel)


def resumes(o: Out) -> None:
    for year, skill, fmt in (
        (2019, "Excel VBA macros", "pdf"),
        (2020, "Power BI", "pdf"),
        (2023, "Apache Spark", "docx"),
    ):
        text = [f"Sam Rivera - Data Analyst ({year})", "Experience at Riverton Retail and Brightline Logistics.",
                f"Highlight: built reporting with {skill}."]  # fmt: skip
        rel = f"Career/Old resumes/resume-{year}.{fmt}"
        (pdf(o.path(rel), ["\n\n".join(text)]) if fmt == "pdf" else docx(o.path(rel), text))
    o.ask("the resume where I listed Spark", "question", "Career/Old resumes/resume-2023.docx")
    o.ask("my resume from 2019", "describe", "Career/Old resumes/resume-2019.pdf")


MEETINGS = [
    ("Pricing review", "raise the delivery fee to $6.99 in March"),
    ("Hiring sync", "open two driver positions in Albany"),
    ("Portal launch", "launch the customer portal on July 29"),
    ("Budget planning", "cut the marketing budget by ten percent"),
    ("Vendor selection", "switch the label printer vendor to PrintCo"),
    ("Security review", "require two-factor login for all staff"),
    ("Office move", "move the Springfield office to the fourth floor"),
    ("Holiday peak", "hire twelve seasonal workers in November"),
    ("Data warehouse", "migrate the warehouse to Snowflake by Q2"),
    ("Customer churn", "call every customer who cancelled in the last 90 days"),
]  # fmt: skip


def meeting_notes(o: Out) -> None:
    rnd = random.Random(105)
    for i in range(30):
        title, decision = MEETINGS[i % len(MEETINGS)]
        date = f"2024-{(i % 12) + 1:02d}-{rnd.randint(1, 28):02d}"
        dec = decision if i < len(MEETINGS) else rnd.choice(["no decision yet, revisit next week",
                                                               "gather more data first"])  # fmt: skip
        rel = f"Work/Meeting notes/{date}-{title.lower().replace(' ', '-')}.md"
        o.path(rel).write_text(f"# {title} - {date}\n\nAttendees: Maria, Sam, Devon, Priya\n\n## Decision\n- We "
                               f"agreed to {dec}.\n\n## Next steps\n- Follow up in two weeks\n")  # fmt: skip
        if i < len(MEETINGS) and i % 2 == 0:
            o.ask(
                f"meeting where we decided to {decision.split(' by ')[0].split(' in ')[0]}",
                "describe",
                rel,
            )


def work_files(o: Out) -> None:
    topics = [
        ("hiring plan", "xlsx", [["Role", "Team", "Quarter"], ["Driver", "Albany", "Q3"], ["Dispatcher", "Springfield", "Q4"]]),
        ("OKRs for 2024", "xlsx", [["Objective", "Key result"], ["Faster deliveries", "on-time rate 97%"]]),
        ("vendor comparison for label printers", "ods", [["Vendor", "Price", "Support"], ["PrintCo", "4200", "24/7"], ["LabelMax", "3900", "weekdays"]]),
        ("warehouse inventory", "xlsx", [["SKU", "Item", "Count"], ["A-100", "pallet wrap", "340"], ["A-220", "cardboard boxes", "1200"]]),
        ("conference travel budget", "ods", [["Item", "Cost"], ["Flights to Denver", "1840"], ["Hotel 3 nights", "960"]]),
        ("customer satisfaction survey results", "xlsx", [["Question", "Score"], ["Delivery speed", "4.3"], ["Driver courtesy", "4.7"]]),
    ]  # fmt: skip
    for name, ext, rows in topics:
        rel = f"Work/Spreadsheets/{name.replace(' ', '-')}.{ext}"
        (xlsx if ext == "xlsx" else ods)(o.path(rel), {name.title()[:30]: rows})
        o.ask(f"spreadsheet with our {name}", "describe", rel)
    decks = [
        ("new employee onboarding", [["Welcome to Brightline"], ["Day 1: badge, laptop, safety training"], ["Who to ask: People team"]]),
        ("customer portal demo", [["Customer portal"], ["Track deliveries live"], ["Download invoices"]]),
        ("safety training for drivers", [["Safe driving"], ["Pre-trip inspection checklist"], ["Report every incident within 24 hours"]]),
        ("annual company kickoff", [["Kickoff 2024"], ["Three priorities: speed, safety, service"]]),
    ]  # fmt: skip
    for name, slides in decks:
        rel = f"Work/Slides/{name.replace(' ', '-')}.pptx"
        pptx(o.path(rel), slides, title=name.title())
        o.ask(f"slides for the {name}", "describe", rel)


# -- notes and recipes --------------------------------------------------------------------------

NOTE_FACTS = [
    ("plumber", "Plumber: Rick at FlowRight Plumbing, 555-0142, fixed the kitchen sink in March."),
    ("electrician", "Electrician Dana Ortiz, 555-0177, installed the porch light."),
    ("locker combination", "Gym locker combination is 24-11-37 (blue locker, row C)."),
    ("Atomic Habits", "Book notes - Atomic Habits: make it obvious, attractive, easy and satisfying."),
    ("Deep Work", "Book notes - Deep Work: schedule focus blocks, quit social media for 30 days."),
    ("food truck idea", "Business idea: a breakfast food truck near the train station selling burritos."),
    ("podcast idea", "Podcast idea: interviews with local small business owners, 20 minutes each."),
    ("garage sale", "Garage sale on June 8: price old bikes at $40, kids clothes $2 each."),
    ("workout plan", "Workout: Monday squats 3x8, Wednesday deadlift 3x5, Friday bench 3x8."),
    ("marathon training", "Marathon training week 6: long run 14 miles on Sunday, easy pace."),
    ("anniversary", "Anniversary ideas: dinner at the rooftop place, book the cabin at Lake Mira."),
    ("babysitter", "Babysitter: Emma, $18/hour, available Fridays, number 555-0199."),
    ("passwords hint", "Reminder: the password manager master hint is our first street name."),
    ("car registration", "Car registration renewal due July 31, cost $96, renew online at the DMV site."),
    ("neighbors", "Neighbors: the Parks at number 44 water our plants when we travel."),
    ("Wi-Fi router", "Router admin page is 192.168.1.1, user admin, reset button on the back."),
    ("spanish class", "Spanish class: Tuesdays 6 pm at the library, teacher Lucia, bring workbook."),
    ("wine list", "Wines we liked: 2019 Rioja from the Italian place, Sancerre at Alex's birthday."),
    ("doctor questions", "Questions for Dr. Patel: knee pain when running, vitamin D dosage."),
    ("moving boxes", "Moving boxes: kitchen boxes are numbered 1-12, books 13-20."),
    ("tea recipe", "Chai: simmer cinnamon, cardamom, ginger and black tea in milk for 8 minutes."),
    ("bike repair", "Bike shop says the rear derailleur needs replacing, quote $85."),
    ("solar panels", "Solar quote from SunPeak: 7.2 kW system, $18,400 before tax credit."),
    ("school pickup", "School pickup rota: Monday Sam, Tuesday Alex, Wednesday the Parks."),
    ("birthday dates", "Birthdays: Jordan Nov 3, Mom May 22, Priya Feb 9."),
]  # fmt: skip
FILLER = ["Call back later.", "Remember to check the calendar.", "Buy more coffee filters.", "Look into this again in spring.",
          "Ask Alex what she thinks.", "Write down the receipt number.", "Maybe next month.", "Done, crossed off."]  # fmt: skip


def notes(o: Out) -> None:
    rnd = random.Random(106)
    for i in range(150):
        ext = "md" if i % 2 else "txt"
        if i < len(NOTE_FACTS):
            _key, fact = NOTE_FACTS[i]
            body = fact + "\n" + " ".join(rnd.sample(FILLER, 2))
            rel = f"Notes/note-{1000 + i}.{ext}"
            o.path(rel).write_text(body + "\n")
            continue
        body = "\n".join(rnd.sample(FILLER, 3))
        o.path(f"Notes/note-{1000 + i}.{ext}").write_text(f"Note {i}\n{body}\n")
    asks = [("the note with the plumber's number", 0), ("who was the electrician that did the porch light", 1),
            ("my gym locker combination", 2), ("my notes on Atomic Habits", 3), ("the food truck business idea", 5),
            ("what are the prices for the garage sale", 7), ("marathon long run plan", 9),
            ("babysitter's phone number", 11), ("when is the car registration due", 13),
            ("how to get into the router settings", 15), ("solar panel quote", 22),
            ("who picks up the kids on Tuesday", 23), ("when is Priya's birthday", 24)]  # fmt: skip
    for q, idx in asks:
        o.ask(q, "question", f"Notes/note-{1000 + idx}.{'md' if idx % 2 else 'txt'}")


RECIPES = [
    ("shakshuka", "eggs poached in tomato and pepper sauce with cumin and feta"),
    ("pad thai", "rice noodles, tamarind, fish sauce, peanuts, bean sprouts"),
    ("beef stew", "chuck roast, carrots, potatoes, red wine, thyme"),
    ("lemon bars", "shortbread crust, lemon curd, powdered sugar"),
    ("vegetable lasagna", "zucchini, spinach, ricotta, lasagna sheets"),
    ("chicken tikka masala", "yogurt marinade, garam masala, tomato cream sauce"),
    ("french onion soup", "caramelized onions, beef broth, gruyere toast"),
    ("blueberry muffins", "fresh blueberries, buttermilk, streusel topping"),
    ("fish tacos", "cod, cabbage slaw, chipotle mayo, corn tortillas"),
    ("mushroom risotto", "arborio rice, porcini, parmesan, white wine"),
    ("pumpkin pie", "pumpkin puree, evaporated milk, cinnamon, nutmeg"),
    ("falafel", "dried chickpeas, parsley, cumin, tahini sauce"),
    ("gazpacho", "cold tomato soup with cucumber, bell pepper, sherry vinegar"),
    ("carrot cake", "grated carrots, walnuts, cream cheese frosting"),
    ("pho", "beef bones, star anise, rice noodles, basil, lime"),
    ("ratatouille", "eggplant, zucchini, peppers, tomatoes, herbes de Provence"),
    ("apple crumble", "baking apples, oats, brown sugar, butter"),
    ("tom yum soup", "lemongrass, galangal, kaffir lime leaves, shrimp"),
    ("chocolate chip cookies", "brown butter, dark chocolate chunks, sea salt"),
    ("paella", "saffron rice, chicken, chorizo, mussels"),
]  # fmt: skip


def recipes(o: Out) -> None:
    made: dict[str, list[str]] = {}
    for i, (dish, ingredients) in enumerate(RECIPES * 2):
        variant = "" if i < len(RECIPES) else " (quick weeknight version)"
        text = f"{dish.title()}{variant}\n\nIngredients: {ingredients}.\nMethod: prepare, cook, taste and adjust."
        rel = f"Recipes/{dish.replace(' ', '-')}{'-quick' if variant else ''}"
        kind = i % 5
        if kind == 0:
            o.path(rel + ".md").write_text(text)
            rel += ".md"
        elif kind == 1:
            o.path(rel + ".txt").write_text(text)
            rel += ".txt"
        elif kind == 2:
            pdf(o.path(rel + ".pdf"), [text])
            rel += ".pdf"
        elif kind == 3:
            docx(o.path(rel + ".docx"), text.split("\n\n"))
            rel += ".docx"
        else:
            scanned_pdf(
                o.path(rel + "-scan.pdf"), [text.replace("\n\n", "\n").split("\n")], seed=400 + i
            )
            rel += "-scan.pdf"
        made.setdefault(dish, []).append(rel)
    # Both versions of a dish are a right answer.
    for i, (dish, _ingredients) in enumerate(RECIPES):
        if i % 2 == 0:
            o.ask(f"recipe for {dish}", "describe", *made[dish])
    o.ask("the recipe that uses tamarind", "question", *made["pad thai"])
    o.ask("soup with lemongrass", "question", *made["tom yum soup"])


# -- health, school, travel, car ------------------------------------------------------------


def health(o: Out) -> None:
    for year, chol, vitd in ((2021, 188, 31), (2022, 197, 24), (2023, 205, 21)):
        pdf(o.path(f"Health/Labs/lab-results-{year}.pdf"), [
            f"SPRINGFIELD MEDICAL LAB\nPatient: Sam Rivera\nCollected: February {year}\nTotal cholesterol: "
            f"{chol} mg/dL\nVitamin D: {vitd} ng/mL\nA1C: 5.{year % 10} %"])  # fmt: skip
    o.ask("blood work from 2022", "describe", "Health/Labs/lab-results-2022.pdf")
    rx = [("lisinopril 10 mg", "blood pressure"), ("cetirizine 10 mg", "seasonal allergies"),
          ("amoxicillin 500 mg", "ear infection"), ("albuterol inhaler", "asthma"),
          ("ibuprofen 600 mg", "knee pain after physical therapy")]  # fmt: skip
    for i, (drug, why) in enumerate(rx):
        rel = f"Health/Prescriptions/prescription-{i + 1}.pdf"
        pdf(o.path(rel), [f"PRESCRIPTION\nPatient: {'Maya' if i == 2 else 'Sam'} Rivera\nMedication: {drug}\n"
                          f"For: {why}\nRefills: {i}\nDr. Anika Patel, Springfield Family Medicine"])  # fmt: skip
        o.ask(f"prescription for the {why}", "question", rel)
    scanned_pdf(o.path("Health/eye-exam-glasses-prescription.pdf"), [[
        "**VISION CENTER - EYEGLASS PRESCRIPTION", "Patient: Sam Rivera",
        "OD: -1.75 sphere, -0.50 cylinder", "OS: -2.00 sphere", "PD: 63 mm"]], seed=501)  # fmt: skip
    o.ask("my glasses prescription", "visual", "Health/eye-exam-glasses-prescription.pdf")
    docx(o.path("Health/allergy-test-results.docx"), [
        "# Allergy test results - Maya Rivera", "Positive: cat dander, birch pollen.", "Negative: peanuts, eggs, dust mites."])  # fmt: skip
    o.ask("is Maya allergic to cats", "question", "Health/allergy-test-results.docx")


def school(o: Out) -> None:
    events = ["picture day on September 20", "book fair the week of October 14", "Halloween parade on October 31",
              "winter concert on December 12", "science fair on February 7", "spring break March 24 to 28",
              "field day on May 30", "last day of school June 13"]  # fmt: skip
    for i, event in enumerate(events):
        month = event.split(" on ")[-1].split(" ")[0] if " on " in event else "March"
        pdf(o.path(f"Kids/Newsletters/oak-hill-newsletter-{i + 1:02d}.pdf"), [
            f"OAK HILL ELEMENTARY - FAMILY NEWSLETTER\nThis month: {event}.\nReminder: label all water bottles."])  # fmt: skip
        if i % 2 == 0:
            o.ask(f"school newsletter about the {event.split(' on ')[0].split(' the week')[0]}", "describe",
                  f"Kids/Newsletters/oak-hill-newsletter-{i + 1:02d}.pdf")  # fmt: skip
        _ = month
    for term, (reading, math) in {
        "fall": ("Meets", "Meets"),
        "winter": ("Meets", "Exceeds"),
    }.items():
        scanned_pdf(o.path(f"Kids/report-card-{term}.pdf"), [[
            "**OAK HILL ELEMENTARY - REPORT CARD", f"Student: Maya Rivera   {term.title()} term",
            f"Reading: {reading} expectations", f"Math: {math} expectations"]], seed=600 + len(term))  # fmt: skip
    o.ask("Maya's winter report card", "visual", "Kids/report-card-winter.pdf")
    for trip in ("aquarium", "apple orchard", "fire station"):
        rel = f"Kids/permission-slip-{trip.replace(' ', '-')}.docx"
        docx(
            o.path(rel),
            [
                f"# Field trip: {trip}",
                "Bus leaves at 9:00. Bring a packed lunch.",
                "Parent signature: ________",
            ],
        )
        o.ask(f"permission slip for the {trip} trip", "describe", rel)


TRIPS = [
    ("Lisbon", "Portugal", "2024-08-03", "Casa do Rio"), ("Denver", "Colorado", "2023-02-17", "Alpine Lodge Hotel"),
    ("Montreal", "Canada", "2023-07-01", "Hotel Le Vieux Port"), ("Tokyo", "Japan", "2022-10-08", "Sakura Stay Shinjuku"),
    ("New Orleans", "Louisiana", "2024-03-01", "Magnolia House"), ("Barcelona", "Spain", "2022-05-20", "Hostal Gotic"),
    ("Yellowstone", "Wyoming", "2023-06-12", "Old Faithful Inn"), ("San Diego", "California", "2024-11-25", "Seaside Suites"),
]  # fmt: skip


def travel(o: Out) -> None:
    rnd = random.Random(107)
    for city, region, day, hotel in TRIPS:
        slug = city.lower().replace(" ", "-")
        ref = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        flight, stay = f"Travel/{slug}/flight-confirmation.pdf", f"Travel/{slug}/hotel-booking.html"
        if city in HAND_WRITTEN_TRIPS:
            flight, stay = HAND_WRITTEN_TRIPS[city]
        else:
            pdf(o.path(flight), [
                f"FLIGHT CONFIRMATION\nBooking reference: {ref}\nBoston to {city}, {region}\nDeparture: {day}\n"
                "Passengers: Sam, Alex and Maya Rivera"])  # fmt: skip
            o.path(stay).write_text(
                f"<html><head><title>Booking confirmed - {hotel}</title></head><body><h1>{hotel}</h1>"
                f"<p>Check-in {day}. {city}, {region}. Breakfast included.</p></body></html>")  # fmt: skip
        docx(o.path(f"Travel/{slug}/itinerary.docx"), [
            f"# {city} itinerary", f"Day 1: arrive and walk around {city}.",
            f"Day 2: {rnd.choice(['museum day', 'food tour', 'hiking', 'boat trip', 'day at the beach'])}."])  # fmt: skip
        o.ask(f"our flight to {city}", "describe", flight)
        o.ask(f"which hotel did we stay at in {city}", "question", stay)


def car_service(o: Out) -> None:
    repairs = [("replaced the timing belt", 640), ("new battery", 189), ("fixed the AC compressor", 1120),
               ("replaced rear brake rotors", 410), ("windshield crack repair", 95), ("check engine light: oxygen sensor", 265)]  # fmt: skip
    for i, (work, cost) in enumerate(repairs):
        rel = f"Car/service-invoice-{i + 1}.pdf"
        if i % 2:
            scanned_pdf(o.path(rel), [[f"**MAIN STREET AUTO - INVOICE {7100 + i}", "Vehicle: 2019 Honda Civic",
                                       f"Work: {work}", f"Total: ${cost}.00"]], seed=700 + i)  # fmt: skip
        else:
            pdf(o.path(rel), [f"MAIN STREET AUTO - INVOICE {7100 + i}\nVehicle: 2019 Honda Civic\nWork done: {work}\n"
                              f"Total: ${cost}.00"])  # fmt: skip
        o.ask(f"invoice from when the {work.split(':')[-1].strip()}"[:70], "describe", rel)


def ebooks(o: Out) -> None:
    books = [("Sourdough at Home", "baking bread with a starter", "feed the starter twice a day"),
             ("The Calm Parent", "raising kids without yelling", "name the feeling before the rule"),
             ("Small Space Gardening", "growing vegetables on a balcony", "use deep pots for tomatoes"),
             ("Investing for Beginners", "index funds and retirement accounts", "low fees matter more than picking stocks"),
             ("Trail Running Basics", "running on mountain trails", "shorten your stride on climbs")]  # fmt: skip
    for title, topic, tip in books:
        rel = f"Books/{title.replace(' ', '-').lower()}.epub"
        w.epub(
            o.path(rel),
            title,
            "Various",
            [("Introduction", f"This book is about {topic}."), ("Chapter 1", f"Tip: {tip}.")],
        )
        o.ask(f"ebook about {topic.split(' with ')[0].split(' on ')[0]}", "describe", rel)


def screenshots(o: Out) -> None:
    shots = [
        ("Delivery", ["Your package was delivered", "Left at the front porch", "Order #48-7721: running shoes"], "package delivered to the porch"),
        ("Calendar", ["Parent-teacher conference", "Thursday 4:15 PM", "Room 12, Ms. Okafor"], "parent teacher conference time"),
        ("Maps", ["24 min to Riverside Park", "via Route 9", "Light traffic"], "directions to Riverside Park"),
        ("Tickets", ["Springfield Symphony", "Saturday 7:30 PM", "Row F, seats 11-12"], "symphony tickets"),
        ("Error", ["Printer offline", "Error code E-204", "Check the network connection"], "printer error code"),
        ("Messages", ["Mom", "Flight lands at 6:40, terminal B", "Don't forget the car seat"], "when does mom's flight land"),
        ("Wallet", ["Gift card balance", "Bookworm Books", "$37.50 remaining"], "bookstore gift card balance"),
        ("Bank", ["Fraud alert", "Did you spend $899.00 at ElectroMart?", "Reply YES or NO"], "the fraud alert from the bank"),
        ("Weather", ["Winter storm warning", "8-12 inches of snow", "Tuesday night"], "snowstorm warning"),
        ("Order", ["Pizza Palace", "Order ready at 6:45", "Large pepperoni, garlic knots"], "pizza order confirmation"),
    ]  # fmt: skip
    for i, (app, lines, query) in enumerate(shots):
        rel = f"Pictures/Screenshots/Screenshot_2024-{(i % 12) + 1:02d}-{i + 3:02d}-{10 + i}{i * 7 % 60:02d}.png"
        w.screenshot(
            app, lines, accent=[(0, 122, 255), (88, 86, 214), (52, 199, 89), (255, 149, 0)][i % 4]
        ).save(o.path(rel))
        o.ask(query, "visual", rel)


# -- handwriting, recordings, videos ----------------------------------------------------------


def handwritten(o: Out, fontfiles: dict) -> None:
    import handwriting as hw

    pages = [
        ("grocery list with oat milk and parmesan", ["Groceries", "eggs (dozen)", "oat milk", "spinach + mushrooms", "parmesan", "dish soap"]),
        ("my handwritten to-do list about the gutters", ["To do this weekend", "clean the gutters", "fix squeaky door", "return library books", "call grandma"]),
        ("notes from the chemistry lecture", ["Chem 101 - Oct 3", "Avogadro's number 6.022 x 10^23", "molar mass = grams per mole", "quiz on Friday"]),
        ("phone numbers I wrote down for the contractor", ["Contractors", "Kitchen: Bianca 555-0123", "Roof: Leo 555-0168", "Painter: Sun 555-0190"]),
        ("grandma's handwritten pancake recipe", ["Grandma's pancakes", "2 cups flour, 2 eggs", "1.5 cups buttermilk", "pinch of nutmeg!", "cook on medium heat"]),
        ("brainstorm for the bakery name", ["Bakery names?", "Rise & Shine", "The Flour Pot", "Crumb Club", "Knead to Know"]),
        ("packing list for the camping trip", ["Camping - pack", "tent + stakes", "sleeping bags x3", "bug spray", "marshmallows"]),
        ("wedding guest list", ["Wedding guests", "the Parks (2)", "Jordan + date", "Priya", "Aunt Rosa (3)"]),
    ]  # fmt: skip
    for i, (query, lines) in enumerate(pages):
        rel = f"Pictures/Notes/IMG_{3100 + i}.jpg"
        hw.notebook_page(lines, fontfiles, 800 + i).save(o.path(rel), quality=85)
        o.ask(query, "visual", rel)
    boards = [
        ("whiteboard with the Q3 roadmap", ["Q3 roadmap", "- customer portal: July 29", "- hire 2 drivers", "- Albany warehouse"]),
        ("the whiteboard from the sprint retro", ["Sprint 14 retro", "- went well: fewer bugs", "- improve: code review speed", "- action: pair on deploys"]),
        ("whiteboard drawing of the database design", ["Database design", "- customers table", "- orders -> customers", "- index on created_at"]),
        ("whiteboard where we split the chores", ["Chores", "- Maya: feed the cat", "- Sam: dishes Mon-Wed", "- Alex: laundry"]),
        ("whiteboard with the marketing plan", ["Marketing plan", "- spring email campaign", "- referral bonus $20", "- local radio ad"]),
    ]  # fmt: skip
    for i, (query, lines) in enumerate(boards):
        rel = f"Pictures/Whiteboard/IMG_{3200 + i}.jpg"
        hw.whiteboard(lines, fontfiles, 900 + i).save(o.path(rel), quality=85)
        o.ask(query, "visual", rel)
    stickies = [
        ("sticky notes with the wifi password for guests", ["wifi: Rivera-Guest", "pw: sunflower-river-42", "router in hallway"]),
        ("sticky notes about the dentist and plumber", ["call plumber 555-0142", "dentist Tue 3:30", "Maya swim bag"]),
        ("sticky note reminding me to renew the passport", ["RENEW PASSPORT before Nov!", "photo 2x2", "fee $130"]),
    ]  # fmt: skip
    for i, (query, notes_) in enumerate(stickies):
        rel = f"Pictures/Notes/IMG_{3300 + i}.jpg"
        hw.sticky_notes(notes_, fontfiles, 950 + i).save(o.path(rel), quality=85)
        o.ask(query, "visual", rel)


MEMOS = [
    ("voice note about buying a new mattress", "Thinking about the new mattress. The firm one at Dream Store was nine hundred dollars, the hybrid was twelve hundred."),
    ("recording where I listed what to pack for Maya's sleepover", "For Maya's sleepover pack her sleeping bag, toothbrush, pajamas and the stuffed rabbit."),
    ("voice memo with the mechanic's advice", "The mechanic said the brakes have about five thousand miles left and the tires are fine."),
    ("my idea for the garden fence", "Idea for the garden fence: use cedar boards, four feet high, with a gate near the shed."),
    ("voice note about the book club pick", "Book club pick for next month is The Midnight Library, we meet at Priya's on the third."),
    ("reminder about the vet appointment", "Reminder: the cat's vet appointment is Thursday at ten, bring the vaccination papers."),
    ("recording of the directions to the cabin", "To get to the cabin, take Route 9 north, turn left after the red barn, the cabin is the second driveway."),
    ("voice note about the speech for Jordan's wedding", "Toast for Jordan's wedding: start with the camping story, thank the parents, keep it under three minutes."),
]  # fmt: skip


def recordings(o: Out) -> None:
    if not w.has_flite():
        return
    for i, (query, text) in enumerate(MEMOS):
        rel = f"Voice/New Recording {i + 4}.m4a"
        w.speech(o.path(rel), text)
        o.ask(query, "spoken", rel)
    talks = [("video explaining how to change the furnace filter", "Changing the furnace filter",
              ["Turn off the furnace", "Slide out the old filter", "Arrow points to the blower"],
              "First turn off the furnace. Slide out the old filter and put in the new one with the arrow pointing toward the blower."),
             ("video tour of the apartment for the subletter", "Apartment tour",
              ["Kitchen", "Bedroom", "Balcony"],
              "This is the kitchen, the dishwasher is new. Here is the bedroom, and the balcony faces the park."),
             ("video about the new expense report process", "Expense reports",
              ["Snap the receipt", "Submit by the 5th", "Manager approves"],
              "Take a photo of every receipt, submit the expense report by the fifth of the month, and your manager approves it.")]  # fmt: skip
    for i, (query, title, bullets, text) in enumerate(talks):
        rel = f"Videos/recording-{i + 10}.mp4"
        w.speech(o.path(rel), text, picture=w.slide(title, bullets))
        o.ask(query, "spoken", rel)


ALL = [bank_statements, utility_bills, paystubs, tax_forms, receipts, job_postings, resumes, meeting_notes,
       work_files, notes, recipes, health, school, travel, car_service, ebooks, screenshots, recordings]  # fmt: skip


def receipt_stores(o: Out) -> dict[str, list[str]]:
    return o.receipt_stores


# -- more of what piles up: manuals, order invoices, letters, contracts, drafts, homework ----

APPLIANCES = [
    ("robot vacuum", "reset the robot vacuum", "hold the home button for 10 seconds until the light blinks"),
    ("espresso machine", "descale the espresso machine", "run the descaling solution through the steam wand"),
    ("smart thermostat", "set a vacation schedule on the thermostat", "open Away mode and choose the dates"),
    ("washing machine", "clean the washing machine filter", "open the small door at the bottom front and unscrew the filter"),
    ("dryer", "clean the dryer lint trap", "pull the screen up from the door opening and wipe it"),
    ("refrigerator", "replace the fridge water filter", "twist the filter in the top right corner a quarter turn"),
    ("microwave", "set the clock on the microwave", "press Clock, enter the time, press Start"),
    ("dishwasher drawer", "fix the dishwasher error E15", "unplug it and tilt to drain the base pan"),
    ("garage door opener", "program a new garage remote", "press the Learn button then the remote within 30 seconds"),
    ("baby monitor", "pair the baby monitor camera", "hold the pair button on the camera and the parent unit"),
    ("electric toothbrush", "change the toothbrush mode", "press the mode button below the power button"),
    ("air purifier", "replace the air purifier filter", "open the back panel and slide in the HEPA filter"),
    ("bread maker", "make gluten free bread in the bread maker", "choose program 9 and use the gluten free flour mix"),
    ("treadmill", "lubricate the treadmill belt", "lift the belt edge and apply silicone along the deck"),
    ("ceiling fan", "reverse the ceiling fan direction for winter", "flip the switch on the motor housing"),
    ("home security camera", "reset the doorbell camera", "press the reset hole for 15 seconds"),
    ("instant pot", "do a quick pressure release on the instant pot", "turn the valve to Venting with a spoon"),
    ("printer", "clean the printer heads", "Settings, Maintenance, Head cleaning"),
    ("electric kettle", "remove limescale from the kettle", "boil water with lemon juice and let it sit"),
    ("water heater", "relight the water heater pilot", "turn the knob to Pilot and press the igniter"),
    ("lawn mower", "change the lawn mower oil", "tip it on its side with the air filter up"),
    ("dehumidifier", "empty and clean the dehumidifier tank", "slide out the tank and wash with mild soap"),
    ("bike computer", "pair the bike speed sensor", "Settings, Sensors, Add sensor, spin the wheel"),
    ("hair dryer", "clean the hair dryer filter", "twist off the rear cap and brush the mesh"),
    ("sound bar", "connect the sound bar to the TV with HDMI ARC", "use the HDMI port labeled ARC on both"),
]  # fmt: skip


def manuals(o: Out) -> None:
    for i, (thing, task, how) in enumerate(APPLIANCES):
        rel = f"Manuals/Appliances/{thing.replace(' ', '-')}-manual.pdf"
        pdf(o.path(rel), [
            f"{thing.upper()} - OWNER'S MANUAL\n\nSafety first: read all instructions.\n\nHow to {task}: {how}.\n\n"
            f"Warranty: {1 + i % 3} year(s) limited. Keep your receipt."])  # fmt: skip
        if i % 2 == 0:
            o.ask(f"how do I {task}", "question", rel)
        else:
            o.ask(f"{thing} manual", "describe", rel)


SHOPS = ["Shopwise", "Megamart Online", "QuickCart", "HomeGoods Direct"]
ORDER_ITEMS = ["phone case", "yoga mat", "bluetooth speaker", "kids rain boots", "coffee grinder", "LED desk lamp",
               "electric toothbrush heads", "backpack", "board game", "water bottle", "running socks",
               "phone tripod", "bath towels set", "kitchen scale", "label maker", "humidifier", "bike lights",
               "laptop stand", "dog harness", "picture frames"]  # fmt: skip


def order_invoices(o: Out) -> None:
    rnd = random.Random(108)
    with_item: dict[str, list[str]] = {}
    asked: list[tuple[str, str]] = []
    for i in range(40):
        shop = SHOPS[i % len(SHOPS)]
        items = [ORDER_ITEMS[i % len(ORDER_ITEMS)], *rnd.sample(ORDER_ITEMS, 1)]
        order = f"{rnd.randint(100, 999)}-{rnd.randint(1000000, 9999999)}"
        rel = f"Downloads/invoice-{order}.pdf"
        pdf(o.path(rel), [
            f"{shop.upper()} - ORDER INVOICE\nOrder number: {order}\nOrdered: 2024-{(i % 12) + 1:02d}-{rnd.randint(1, 28):02d}\n"
            + "\n".join(f"{it}  ${rnd.uniform(9, 120):.2f}" for it in items) + "\nShipped to: Sam Rivera, 88 Willow Court"])  # fmt: skip
        for it in set(items):
            with_item.setdefault(it, []).append(rel)
        if i < 20 and i % 3 == 0:
            asked.append((items[0], rel))
        if i == 7:
            o.ask(f"order number {order}", "exact", rel)
    for item, _rel in asked:  # every invoice with that item is a right answer
        o.ask(f"invoice for the {item} I ordered online", "describe", *with_item[item])


LETTERS = [
    ("property tax assessment", "COUNTY ASSESSOR", "Your property at 42 Maple Street is assessed at $361,000 for 2024. You may appeal by May 1."),
    ("insurance claim decision", "HARBOR HOME INSURANCE", "Claim HH-55120 for the hail damage to the roof is approved. Payment: $7,840 minus the $1,000 deductible."),
    ("school bus route change", "SPRINGFIELD SCHOOL DISTRICT", "Starting October 7, bus 14 will pick up at Willow Court and Elm at 7:42 AM."),
    ("library fine notice", "SPRINGFIELD PUBLIC LIBRARY", "Two items are overdue: The Hobbit and Dinosaur Atlas. Fine so far: $3.50."),
    ("homeowners association playground vote", "MAPLE STREET HOA", "Vote on the new playground: ballots due by June 15. Cost per home: $120."),
    ("medical bill", "SPRINGFIELD GENERAL HOSPITAL", "Emergency room visit on March 2 for Maya Rivera. Amount you owe after insurance: $250."),
    ("credit card replacement", "CEDAR CREDIT UNION", "Your new Visa card ending 9920 is enclosed. The old card ending 4471 stops working in 14 days."),
    ("voter registration confirmation", "BOARD OF ELECTIONS", "You are registered to vote at Lincoln School, precinct 7."),
    ("car recall notice", "VEHICLE SAFETY RECALL NOTICE", "Recall 24-112 for the fuel pump on your 2019 Civic. Repair is free at any dealer."),
    ("rent receipt from landlord", "LAKESIDE PROPERTY MANAGEMENT", "Received $1,850 for October rent, unit 4B. Thank you."),
]  # fmt: skip


def letters(o: Out) -> None:
    for i, (what, sender, body) in enumerate(LETTERS * 2):
        rel = f"Scans/scan_{8000 + i:04d}.pdf"
        if i < len(LETTERS):
            scanned_pdf(o.path(rel), [[f"**{sender}", "Dear Sam Rivera,", *[body[j:j + 52] for j in range(0, len(body), 52)]]],
                        seed=1000 + i)  # fmt: skip
            o.ask(f"the letter about the {what}", "visual", rel)
        else:
            scanned_pdf(
                o.path(rel),
                [[f"**{sender}", "Reminder", "Please keep this notice for your records."]],
                seed=1000 + i,
            )


def contracts(o: Out) -> None:
    deals = [("gym membership", "IronWorks Gym: $39/month, 12-month minimum, cancel with 30 days notice in writing."),
             ("cell phone contract", "Northstar Mobile: device payment plan for 3 phones, 24 months, $28/month each."),
             ("freelance contract with Lumen Labs", "Sam Rivera will build a sales dashboard for $6,500, due by August 30."),
             ("car lease", "Lease of a 2022 Toyota RAV4 for 36 months, 10,000 miles a year, $389/month."),
             ("daycare enrollment agreement", "Little Acorns Daycare: Monday to Friday, 8 am to 5:30 pm, $1,350/month."),
             ("lawn care service agreement", "GreenCut: mowing every other week April to October, $45 per visit."),
             ("home security monitoring contract", "SafeHome monitoring: $24.99/month, 36-month term, includes 2 cameras."),
             ("storage unit rental", "Unit 118 at StoreMore, 10x10 ft, $129/month, month to month."),
             ("music lessons agreement", "Weekly piano lessons for Maya with Ms. Alvarez, $35 per 30 minutes."),
             ("solar panel lease", "SunPeak solar lease: 20 years, $109/month, panels stay the property of SunPeak.")]  # fmt: skip
    for i, (what, body) in enumerate(deals):
        rel = f"Documents/Contracts/{what.replace(' ', '-')}.{'pdf' if i % 2 == 0 else 'docx'}"
        (
            pdf(o.path(rel), [what.upper() + "\n\n" + body])
            if i % 2 == 0
            else docx(o.path(rel), [f"# {what.title()}", body])
        )
        o.ask(f"how long is the {what.split(' with ')[0]}" if i % 3 == 0 else f"{what.split(' with ')[0]} contract",
              "question" if i % 3 == 0 else "describe", rel)  # fmt: skip


def drafts_and_homework(o: Out) -> None:
    rnd = random.Random(109)
    drafts = [("complaint to the airline about the lost suitcase", "Dear Atlantic Air, my suitcase did not arrive in Lisbon on flight 2210..."),
              ("thank you note to Ms. Okafor", "Dear Ms. Okafor, thank you for helping Maya with her reading this year..."),
              ("letter disputing the parking ticket", "I am writing to dispute citation 44120 because the meter was broken..."),
              ("resignation letter draft", "Dear Maria, please accept this letter as my resignation from Brightline..."),
              ("reference request to my old manager", "Hi Devon, would you be willing to be a reference for the Acme role?..."),
              ("email to the landlord about the leaking faucet", "Hello, the bathroom faucet in unit 4B has been leaking since Monday...")]  # fmt: skip
    for i in range(30):
        if i < len(drafts):
            what, text = drafts[i]
            rel = f"Documents/Drafts/draft-{i + 1}.txt"
            o.path(rel).write_text(text + "\n")
            o.ask(f"my draft {what}", "describe", rel)
        else:
            o.path(f"Documents/Drafts/draft-{i + 1}.txt").write_text(rnd.choice(FILLER) + "\n")
    homework = [("book report on Charlotte's Web", "Charlotte is a spider who saves Wilbur the pig by writing words in her web."),
                ("science project about volcanoes", "We made a volcano with baking soda and vinegar. Hypothesis: more vinegar, bigger eruption."),
                ("essay about my favorite animal", "My favorite animal is the octopus because it has three hearts and can change color."),
                ("math worksheet on fractions", "1/2 + 1/4 = 3/4. 2/3 of 9 = 6. Which is bigger, 3/5 or 2/3?"),
                ("spelling list week 12", "because, friend, laugh, through, enough, beautiful")]  # fmt: skip
    for i in range(15):
        what, text = homework[i % len(homework)]
        rel = f"Kids/Homework/maya-homework-{i + 1}.docx"
        docx(
            o.path(rel),
            [
                f"# {what.title()}" if i < len(homework) else f"# Homework {i + 1}",
                text if i < len(homework) else "Practice page.",
            ],
        )
        if i < len(homework):
            o.ask(f"Maya's {what}", "describe", rel)


ALL += [manuals, order_invoices, letters, contracts, drafts_and_homework]
