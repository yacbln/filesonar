"""Write the benchmark's mock files: the kind of things people keep on their computers.

    python benchmark/make_corpus.py OUT_DIR

About 75 files for one made-up person, Sam Rivera: insurance, housing, taxes, bank and
pay papers, receipts (photos), warranties and manuals, résumés, work slides and
spreadsheets, health, school, travel, recipes, notes, screenshots, a French lease and
bill, an e-book, voice memos and narrated videos. Look-alikes are there on
purpose (car vs home vs renters insurance, two tax years, two résumés, several
receipts), as in real life. Every person, company and number is invented.

The output is deterministic. Voice memos and videos need ffmpeg built with flite (its
built-in speech synthesizer); without it they are skipped with a warning.
"""

from __future__ import annotations

import os
import random
import sys
from datetime import datetime
from pathlib import Path

from writers import (
    docx,
    epub,
    has_flite,
    ods,
    odt,
    pdf,
    photo_of,
    pptx,
    scanned_pdf,
    screenshot,
    slide,
    speech,
    text_image,
    xlsx,
)

# -- the files ------------------------------------------------------------------------------

LEASE = [
    "RESIDENTIAL LEASE AGREEMENT\n\nThis Lease Agreement is made on August 1, 2023 between Lakeside "
    "Property Management (Landlord) and Sam Rivera (Tenant) for the apartment at 88 Willow Court, "
    "Unit 4B, Springfield.\n\n1. TERM. The lease begins on September 1, 2023 and ends on August 31, "
    "2024 (twelve months).\n\n2. RENT. Monthly rent is $1,850, due on the first day of each month. A "
    "late fee of $75 applies after the fifth day.\n\n3. SECURITY DEPOSIT. Tenant pays a security "
    "deposit of $1,850. The deposit is returned within 30 days after move-out, minus the cost of any "
    "damage beyond normal wear and tear.",
    "4. UTILITIES. Tenant pays electricity and internet. Water and trash are included in the rent.\n\n"
    "5. PETS. One cat or one small dog under 25 pounds is allowed with a one-time pet fee of $300.\n\n"
    "6. PARKING. One assigned parking space, number 17, is included.\n\n7. NOTICE. Either party must "
    "give 60 days written notice before the end of the term to end the lease.\n\n"
    "Signed: Sam Rivera, Tenant        Dana Whitfield, Lakeside Property Management",
]


def build(out: Path) -> list[str]:
    """The hand-made core: ~75 files, one of each kind of thing people keep."""
    rnd = random.Random(7)
    made: list[str] = []
    skipped: list[str] = []

    def path(rel: str) -> Path:
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        made.append(rel)
        return p

    # Insurance: three look-alike policies and a card.
    pdf(path("Insurance/auto-policy-2024.pdf"), [
        "NORTHWIND AUTO INSURANCE\nPolicy Declarations\n\nNamed insured: Sam Rivera\nPolicy number: "
        "NW-58213-04\nPolicy period: March 15, 2024 to March 15, 2025\nVehicle: 2019 Honda Civic LX, VIN "
        "QO0BENCHMARK00001\n\nCoverage: Bodily injury liability $100,000/$300,000. Property damage "
        "$50,000. Collision with $500 deductible. Comprehensive with $250 deductible. Roadside "
        "assistance included.\n\nAnnual premium: $1,284.00, paid in monthly installments of $107.00.\n"
        "Your policy renews automatically on March 15, 2025."
    ], title="Auto insurance policy")  # fmt: skip
    pdf(path("Insurance/home-insurance-declarations.pdf"), [
        "HARBOR HOME INSURANCE COMPANY\nHomeowners Policy Declarations\n\nInsured: Sam and Alex Rivera"
        "\nProperty: 42 Maple Street, Springfield\nPolicy number: HH-0093321\nPolicy term: June 1, 2024 "
        "to June 1, 2025\n\nDwelling coverage: $385,000\nPersonal property: $150,000\nLoss of use: "
        "$40,000\nPersonal liability: $300,000\nDeductible (all perils): $1,000\nWind and hail "
        "deductible: 2% of dwelling coverage\n\nAnnual premium: $1,630.00"
    ])  # fmt: skip
    pdf(path("Insurance/renters-insurance-old-apartment.pdf"), [
        "COZY NEST RENTERS INSURANCE\nRenters policy summary\n\nPolicyholder: Sam Rivera\nRental "
        "address: 15 Elm Street, Apartment 2, Riverton\nCoverage period: 2021 to 2022\n\nPersonal "
        "property: $25,000\nLiability: $100,000\nDeductible: $500\nMonthly premium: $14.50\n\nThis "
        "policy was cancelled when the policyholder moved out in August 2022."
    ])  # fmt: skip
    text_image([
        "**BlueRiver Health", "", "Member: SAM RIVERA", "Member ID: BRH 448 902 117",
        "Group number: 77120", "Plan: Silver PPO", "Primary care copay: $25",
        "Specialist copay: $50", "Rx BIN 610014",
    ], width=900, size=34, bg=(222, 236, 250)).save(path("Insurance/health-insurance-card.png"))  # fmt: skip

    # Housing.
    pdf(path("Housing/lease-agreement-2023.pdf"), LEASE, title="Residential Lease Agreement")
    docx(path("Housing/lease-renewal-offer-2024.docx"), [
        "# Lease renewal offer",
        "Dear Sam Rivera,",
        "Your lease for 88 Willow Court, Unit 4B ends on August 31, 2024. We are happy to offer you a "
        "renewal for another twelve months starting September 1, 2024.",
        "The new monthly rent will be $1,925, an increase of $75. All other terms stay the same.",
        "Please sign and return this offer by July 1, 2024. Lakeside Property Management",
    ])  # fmt: skip
    (path("Housing/move-out-checklist.md")).write_text(
        "# Move-out checklist\n\n- [ ] Patch nail holes in the living room\n- [ ] Deep clean the oven and "
        "fridge\n- [ ] Return both keys and the parking fob to the office\n- [ ] Take photos of every room "
        "for the security deposit\n- [ ] Forward mail to the new address\n- [ ] Cancel internet and "
        "transfer electricity\n"
    )
    pdf(path("Housing/electric-bill-march-2024.pdf"), [
        "METRO POWER & LIGHT\nAccount: 3301-772-845   Service address: 88 Willow Court Unit 4B\n"
        "Billing period: February 20 to March 19, 2024\n\nEnergy used: 612 kWh\nEnergy charge: $71.38\n"
        "Delivery charge: $12.90\nTaxes and fees: $3.14\n\nAMOUNT DUE: $87.42 by April 8, 2024\n"
        "Compared with last March you used 8% less electricity."
    ])  # fmt: skip
    scanned_pdf(path("Housing/internet-bill-scan.pdf"), [[
        "**FIBERNET HOME INTERNET", "Invoice 2024-05-118842", "Customer: Sam Rivera",
        "Plan: Fiber 500 Mbps", "Monthly service ........ $65.00", "Equipment (router) ..... $4.99",
        "**Total due: $69.99", "Due date: May 25, 2024", "Thank you for choosing FiberNet",
    ]], seed=11)  # fmt: skip

    # Taxes and money.
    for year, line in (("2023", "Refund amount: $1,142.00, sent by direct deposit on April 2, 2024."),
                       ("2022", "Amount owed: $230.00, paid on April 14, 2023.")):  # fmt: skip
        pdf(path(f"Taxes/{year}-tax-return-summary.pdf"), [
            f"FEDERAL INCOME TAX RETURN SUMMARY - TAX YEAR {year}\n\nFiling status: Married filing jointly\n"
            "Taxpayers: Sam Rivera and Alex Rivera\n\nWages, salaries, tips: see W-2 forms\nStandard "
            f"deduction applied\nChild tax credit: claimed for one dependent\n\n{line}\n\nPrepared with "
            "TaxEase Online. Keep this summary with your records for at least three years."
        ])  # fmt: skip
    scanned_pdf(path("Taxes/w2-2023-scan.pdf"), [[
        "**Form W-2 Wage and Tax Statement 2023", "Employer: Brightline Logistics Inc.",
        "Employer ID (EIN): 47-1188290", "Employee: Sam Rivera",
        "1 Wages, tips, other compensation: 78,450.00", "2 Federal income tax withheld: 9,120.00",
        "4 Social security tax withheld: 4,863.90", "6 Medicare tax withheld: 1,137.53",
    ]], seed=12)  # fmt: skip
    pdf(path("Finance/bank-statement-may-2024.pdf"), [
        "CEDAR CREDIT UNION - Checking account statement\nAccount ending 4471   Statement period: May 1 - "
        "May 31, 2024\n\nOpening balance: $4,212.55\n05/01  Rent payment - Lakeside Property Mgmt  "
        "-1,850.00\n05/03  Payroll deposit - Brightline Logistics  +2,615.40\n05/07  Green Basket Market  "
        "-142.18\n05/12  Metro Power & Light  -87.42\n05/15  Transfer to savings  -400.00\n05/17  Payroll "
        "deposit - Brightline Logistics  +2,615.40\n05/22  Hartley Hardware  -63.95\n05/28  FiberNet  "
        "-69.99\nClosing balance: $6,829.81"
    ])  # fmt: skip
    xlsx(path("Finance/monthly-budget-2024.xlsx"), {
        "Budget": [["Category", "Planned", "Spent"], ["Rent", "1850", "1850"], ["Groceries", "600", "642"],
                   ["Utilities", "180", "157"], ["Car payment", "315", "315"], ["Eating out", "200", "260"],
                   ["Kids activities", "150", "120"]],
        "Savings goals": [["Goal", "Target", "Saved so far"], ["Vacation fund (Lisbon)", "3000", "2150"],
                          ["Emergency fund", "10000", "6400"], ["New laptop", "1400", "300"]],
    })  # fmt: skip
    pdf(path("Finance/paystub-june-2024.pdf"), [
        "BRIGHTLINE LOGISTICS INC. - EARNINGS STATEMENT\nEmployee: Sam Rivera   Pay date: June 14, 2024   "
        "Pay period: May 27 - June 9, 2024\n\nGross pay: $3,153.85\nFederal income tax: -$352.10\nSocial "
        "security: -$195.54\nMedicare: -$45.73\nHealth insurance (BlueRiver Silver PPO): -$118.00\n401(k) "
        "contribution 6%: -$189.23\n\nNet pay: $2,253.25\nYear to date gross: $37,846.20"
    ])  # fmt: skip
    pdf(path("Finance/car-loan-statement.pdf"), [
        "SUMMIT AUTO FINANCE\nLoan statement for: Sam Rivera\nVehicle: 2019 Honda Civic LX\nLoan number: "
        "SAF-220193\n\nMonthly payment: $315.00 due on the 10th\nInterest rate: 4.9% APR\nRemaining "
        "balance: $6,480.22\nPayments remaining: 22\n\nPay online at summitautofinance.example"
    ])  # fmt: skip

    # Receipts, photographed with a phone.
    receipts = [
        ("Receipts/IMG_1043.jpg", ["**HARTLEY HARDWARE", "1200 Main St, Springfield", "05/22/2024 14:31",
         "", "Cordless drill 18V      49.99", "Wood screws 100pk        6.49", "Wall anchors             3.99",
         "", "SUBTOTAL                60.47", "TAX                      3.48", "**TOTAL               63.95",
         "VISA ****4471"]),
        ("Receipts/IMG_1057.jpg", ["**APPLIANCE WORLD", "Springfield Plaza", "04/06/2024 11:02", "",
         "Larkfield 300 dishwasher", "Model LK-300DW         649.00", "Haul-away old unit     25.00",
         "Installation           89.00", "", "**TOTAL              813.00",
         "2-year manufacturer warranty", "Keep this receipt for warranty"]),
        ("Receipts/IMG_1102.jpg", ["**GREEN BASKET MARKET", "Fresh groceries since 1987", "05/07/2024 18:45",
         "", "Chickpeas 2x            2.58", "Coconut milk            2.29", "Basmati rice 2lb        4.99",
         "Bananas                 1.87", "Chicken thighs         11.42", "Milk 1 gal              4.19",
         "**TOTAL              142.18"]),
        ("Receipts/scan_0007.png", ["**LUIGI'S TRATTORIA", "Italian kitchen - 22 Harbor Rd", "06/21/2024 20:14",
         "Table 12   Guests 2", "", "Lasagna della casa      21.00", "Spaghetti carbonara     19.00",
         "Tiramisu                 9.00", "House red wine x2       24.00", "", "**TOTAL               84.00"]),
        ("Receipts/pharmacy-receipt.png", ["**WELLCARE PHARMACY", "Store 0418", "03/11/2024 09:20", "",
         "Rx 7781223 AMOXICILLIN 500MG", "Qty 21   Copay           10.00", "Ibuprofen 200mg          7.49",
         "", "**TOTAL               17.49", "Patient: RIVERA, MAYA"]),
    ]  # fmt: skip
    for i, (rel, lines) in enumerate(receipts):
        img = text_image(lines, width=640, mono=True, size=24, bg=(250, 249, 244))
        img = photo_of(img, 100 + i)
        img.save(path(rel), quality=88) if rel.endswith(".jpg") else img.save(path(rel))

    # Warranties and manuals.
    pdf(path("Manuals/dishwasher-warranty.pdf"), [
        "LARKFIELD HOME APPLIANCES - LIMITED WARRANTY\nModel: LK-300DW dishwasher\n\nThis appliance is covered "
        "for two years from the date of purchase against defects in materials and workmanship. The "
        "stainless steel tub is covered for the lifetime of the product. To make a claim, keep your "
        "proof of purchase and call customer service. Damage from improper installation is not covered."
    ])  # fmt: skip
    path("Manuals/laptop-warranty-registration.html").write_text(
        "<html><head><title>Product registration confirmed - Zenbook support</title><style>body{font:14px}"
        "</style><script>track()</script></head><body><nav>Home | Support | Account</nav><h1>Thanks, Sam! "
        "Your laptop is registered.</h1><p>Product: Zenbook 14 OLED laptop</p><p>Serial number: "
        "N2K8CV044291</p><p>Warranty: 1 year standard, extended to 3 years with ExtendCare until "
        "January 9, 2027.</p></body></html>"
    )
    pdf(path("Manuals/coffee-maker-manual.pdf"), [
        "BREWMASTER 12-CUP COFFEE MAKER - USER MANUAL\n\nFirst use: run one full cycle with water only.\n\n"
        "Cleaning and descaling: when the CLEAN light blinks, mineral deposits have built up. Fill the "
        "reservoir with equal parts white vinegar and water, press CLEAN, and let the cycle finish. Then "
        "run two cycles with fresh water to rinse.\n\nTroubleshooting: if the coffee is weak, use 1 "
        "tablespoon of ground coffee per cup."
    ], title="BrewMaster coffee maker manual")  # fmt: skip
    pdf(path("Manuals/bookshelf-assembly-instructions.pdf"), [
        "OAKWOOD 5-SHELF BOOKCASE - ASSEMBLY INSTRUCTIONS\nParts: 2 side panels (A), 5 shelves (B), back "
        "panel (C), 20 cam locks, 20 dowels, 1 Allen key.\n\nStep 1: Insert dowels into the side panels.\n"
        "Step 2: Attach the shelves with cam locks and turn clockwise.\nStep 3: Nail the back panel.\n"
        "Step 4: Fix the bookcase to the wall with the anti-tip strap. Assembly takes about 45 minutes."
    ])  # fmt: skip

    # Work and career, with an old and a new résumé.
    docx(path("Career/Sam_Rivera_Resume_2024.docx"), [
        "# Sam Rivera", "Data Analyst | Springfield | sam.rivera@example.com",
        "# Experience",
        "Brightline Logistics, Senior Data Analyst, 2021 - present. Built delivery-time dashboards in "
        "Tableau, automated weekly reports with Python and SQL, cut late deliveries by 14%.",
        "Riverton Retail, Data Analyst, 2018 - 2021. Forecasted store demand and analysed promotions.",
        "# Skills", "SQL, Python (pandas), Tableau, Excel, A/B testing, dbt",
    ], title="Resume 2024")  # fmt: skip
    docx(path("Career/Sam_Rivera_Resume_2022_old.docx"), [
        "# Sam Rivera", "Data Analyst | Riverton",
        "# Experience",
        "Brightline Logistics, Data Analyst, 2021 - present. Weekly reporting and SQL queries.",
        "Riverton Retail, Junior Data Analyst, 2018 - 2021. Excel reports for store managers.",
        "# Skills", "SQL, Excel, some Python",
    ])  # fmt: skip
    docx(path("Career/Cover_Letter_Acme_Analytics.docx"), [
        "Dear Hiring Team at Acme Analytics,",
        "I am applying for the Lead Data Analyst role. At Brightline Logistics I turned messy delivery "
        "data into dashboards that operations teams use every morning, and I would love to bring that "
        "experience to Acme's customer insights team.",
        "Thank you for your time. Sincerely, Sam Rivera",
    ])  # fmt: skip
    pdf(path("Career/offer-letter-brightline.pdf"), [
        "BRIGHTLINE LOGISTICS INC.\nMarch 3, 2021\n\nDear Sam,\n\nWe are delighted to offer you the position "
        "of Data Analyst. Your starting annual salary will be $82,000, paid every two weeks. Your start "
        "date is April 5, 2021. You will be eligible for health insurance, a 401(k) plan with a 4% "
        "company match, and 18 days of paid time off per year.\n\nPlease sign below to accept this "
        "offer.\n\nMaria Chen, Head of People"
    ])  # fmt: skip
    docx(path("Career/performance-review-2023.docx"), [
        "# Annual performance review 2023 - Sam Rivera",
        "Overall rating: Exceeds expectations.",
        "Strengths: the delivery dashboard project, clear communication with the operations team.",
        "Goals for 2024: mentor the two new analysts, learn dbt, lead the warehouse forecasting project.",
        "Merit increase: 5% effective January 1, 2024.",
    ])  # fmt: skip
    path("Career/job-posting-lead-data-analyst.txt").write_text(
        "Lead Data Analyst - Acme Analytics (hybrid, Springfield)\n\nAbout the role: lead a team of four "
        "analysts on the customer insights team. Requirements: 5+ years with SQL and Python, experience "
        "with Tableau or Looker, strong stakeholder communication. Salary range $105,000 - $125,000.\n"
    )
    pptx(path("Work/Q3-sales-review.pptx"), [
        ["Q3 sales review", "Brightline Logistics"],
        ["Revenue up 12% compared with Q2", "Northeast region grew fastest: +21%"],
        ["Late deliveries down to 3.1%", "New warehouse in Albany opened in August"],
        ["Next quarter", "Holiday peak staffing", "Launch the customer portal"],
    ], notes={2: "Mention that the Northeast growth came from two new grocery clients."},
       title="Q3 sales review")  # fmt: skip
    xlsx(path("Work/project-timeline.xlsx"), {
        "Timeline": [["Milestone", "Owner", "Due", "Status"],
                     ["Requirements signed off", "Maria", "2024-04-15", "Done"],
                     ["Customer portal beta", "Sam", "2024-06-30", "Late"],
                     ["Customer portal launch", "Sam", "2024-07-29", "Planned"]],
    })  # fmt: skip
    path("Work/meeting-notes-2024-05-14.md").write_text(
        "# Product meeting - May 14, 2024\n\nAttendees: Maria, Sam, Devon, Priya\n\n## Decisions\n- The "
        "customer portal launch moves from June to the end of July because the payment integration is "
        "not ready.\n- Devon owns the payment testing.\n\n## Action items\n- Sam: update the project "
        "timeline\n- Priya: tell the sales team about the new date\n"
    )
    odt(path("Work/team-offsite-agenda.odt"), [
        "Team offsite agenda - Lakeview Lodge, September 12-13",
        "Day 1: 9:00 welcome and coffee, 10:00 strategy for next year, 13:00 lunch by the lake, "
        "15:00 hiking or kayaking.",
        "Day 2: 9:30 workshop on dashboards, 12:00 closing lunch and bus back to the office.",
    ])  # fmt: skip

    # Health.
    pdf(path("Health/blood-test-results-2024.pdf"), [
        "SPRINGFIELD MEDICAL LAB - TEST RESULTS\nPatient: Sam Rivera   Collected: February 6, 2024\n\n"
        "Total cholesterol: 212 mg/dL (high, ref < 200)\nLDL: 138 mg/dL (high)\nHDL: 52 mg/dL\n"
        "Glucose, fasting: 94 mg/dL (normal)\nVitamin D, 25-OH: 18 ng/mL (LOW, ref 30-100)\nTSH: 2.1 "
        "(normal)\n\nDoctor's note: start vitamin D 2000 IU daily and recheck in three months."
    ])  # fmt: skip
    scanned_pdf(path("Health/vaccination-record-scan.pdf"), [[
        "**IMMUNIZATION RECORD", "Name: Maya Rivera   Date of birth: 04/12/2016",
        "DTaP: 06/2016, 08/2016, 10/2016, 07/2017, 05/2020", "MMR: 05/2017, 05/2020",
        "Varicella: 05/2017, 05/2020", "Influenza: every October", "Clinic: Springfield Pediatrics",
    ]], seed=13)  # fmt: skip
    screenshot("Bright Smile Dental", [
        "Appointment confirmed", "Teeth cleaning with Dr. Patel", "Tuesday, July 9 at 3:30 PM",
        "450 Oak Avenue, Suite 2", "Reply C to confirm or R to reschedule",
    ], accent=(0, 150, 136)).save(path("Health/dentist-appointment-confirmation.png"))  # fmt: skip
    docx(path("Health/physical-therapy-knee-exercises.docx"), [
        "# Home exercises for the left knee",
        "1. Straight leg raises: 3 sets of 10, hold for 3 seconds.",
        "2. Wall sits: 3 times 30 seconds.",
        "3. Step-ups on the bottom stair: 2 sets of 12.",
        "Ice for 15 minutes after the exercises. Next physical therapy visit on June 20.",
    ])  # fmt: skip

    # Kids and school.
    pdf(path("Kids/school-supply-list-grade-3.pdf"), [
        "OAK HILL ELEMENTARY - GRADE 3 SUPPLY LIST\n\n24 No. 2 pencils, 2 boxes of crayons, 4 glue sticks, "
        "1 pair of safety scissors, 3 spiral notebooks (wide ruled), 1 box of tissues, 1 pack of "
        "dry-erase markers, headphones in a labelled bag, a backpack without wheels."
    ])  # fmt: skip
    scanned_pdf(path("Kids/report-card-spring.pdf"), [[
        "**OAK HILL ELEMENTARY - REPORT CARD", "Student: Maya Rivera   Grade 2   Spring term",
        "Reading: Meets expectations", "Math: Exceeds expectations", "Science: Meets expectations",
        "Behaviour: Kind and helpful to classmates", "Teacher: Ms. Okafor",
    ]], seed=14)  # fmt: skip
    docx(path("Kids/field-trip-permission-slip.docx"), [
        "# Field trip permission slip",
        "Grade 3 will visit the Springfield Science Museum on Friday, October 18. The bus leaves at "
        "8:45 and returns by 2:30. Cost: $12 for the planetarium show.",
        "I give permission for my child ____________ to attend.  Parent signature: ____________",
    ])  # fmt: skip
    path("Kids/soccer-practice-schedule.csv").write_text(
        "date,time,field,note\n2024-09-07,09:00,Riverside Park Field 3,bring shin guards\n"
        "2024-09-14,09:00,Riverside Park Field 3,team photo day\n2024-09-21,10:30,Lincoln School,"
        "game vs Hornets\n2024-09-28,09:00,Riverside Park Field 3,snack: Rivera family\n"
    )

    # Travel.
    pdf(path("Travel/flight-confirmation-lisbon.pdf"), [
        "ATLANTIC AIR - BOOKING CONFIRMATION\nBooking reference: K7Q2ZP\nPassengers: Sam Rivera, Alex "
        "Rivera, Maya Rivera\n\nOutbound: Boston (BOS) to Lisbon (LIS), Atlantic Air flight 2210, August 3, 2024, "
        "departs 21:40, arrives 09:15 +1\nReturn: Lisbon (LIS) to Boston (BOS), flight 2211, August "
        "17, 2024, departs 12:05\n\nBaggage: one checked bag of 23 kg per passenger. Online check-in "
        "opens 24 hours before departure."
    ])  # fmt: skip
    screenshot("Atlantic Air", [
        "BOARDING PASS", "RIVERA / SAM", "BOS → LIS   Flight AT-2210", "Seat 23C   Group 4",
        "Boarding 20:55   Gate B14",
    ], accent=(196, 30, 58)).save(path("Travel/boarding-pass.png"))  # fmt: skip
    path("Travel/hotel-booking-lisbon.html").write_text(
        "<html><head><title>Your booking at Casa do Rio, Lisbon</title><script>analytics()</script>"
        "</head><body><h1>Booking confirmed</h1><p>Casa do Rio Guesthouse, Rua das Flores, Lisbon"
        "</p><p>Check-in: August 4, 2024 from 14:00. Check-out: August 17, 2024 by 11:00.</p><p>Family "
        "room with balcony, breakfast included. Total: EUR 1,690.</p></body></html>"
    )
    pdf(path("Travel/passport-renewal-form.pdf"), [
        "APPLICATION FOR PASSPORT RENEWAL BY MAIL\n\nUse this form if your most recent passport was issued "
        "when you were 16 or older, within the last 15 years, and is undamaged. Include your most "
        "recent passport, one passport photo (2 x 2 inches), and the fee of $130.\n\nApplicant: Sam "
        "Rivera   Current passport number: 562118904   Expiration date: 11/2024"
    ])  # fmt: skip
    path("Travel/packing-list.txt").write_text(
        "Packing list - Lisbon\n\npassports!!\nphone chargers + EU plug adapter\nsunscreen, hats\n"
        "comfortable walking shoes\nMaya's swimsuit and goggles\nmotion sickness tablets\nprinted "
        "hotel booking\n"
    )
    docx(path("Travel/rome-itinerary-2022.docx"), [
        "# Rome trip, May 2022",
        "Day 1: Colosseum and Roman Forum (tickets booked for 10:00).",
        "Day 2: Vatican Museums and St. Peter's Basilica.",
        "Day 3: Trastevere food tour, gelato at Fatamorgana.",
    ])  # fmt: skip

    # Recipes and home.
    path("Recipes/chickpea-curry.md").write_text(
        "# Easy chickpea curry\n\nServes 4.\n\n## Ingredients\n- 2 cans chickpeas\n- 1 can coconut milk\n"
        "- 1 onion, 3 garlic cloves, thumb of ginger\n- 2 tbsp curry powder, 1 can chopped tomatoes\n\n"
        "## Steps\n1. Fry the onion, garlic and ginger.\n2. Add the curry powder, then tomatoes and "
        "coconut milk.\n3. Add the chickpeas and simmer 20 minutes. Serve with basmati rice.\n"
    )
    path("Recipes/grandmas-banana-bread.txt").write_text(
        "Grandma June's banana bread\n\n3 very ripe bananas, 1/3 cup melted butter, 3/4 cup sugar, 1 egg, "
        "1 tsp vanilla, 1 tsp baking soda, pinch of salt, 1 1/2 cups flour. Mash the bananas, mix "
        "everything, bake at 350F for 60 minutes. Grandma always added a handful of walnuts.\n"
    )
    scanned_pdf(path("Recipes/lasagna-recipe-scan.pdf"), [[
        "**Nonna Rosa's lasagna", "1 lb ground beef, 1 lb Italian sausage", "1 jar marinara sauce,",
        "15 oz ricotta, 1 egg, parsley", "2 cups shredded mozzarella", "12 lasagna noodles",
        "Layer and bake at 375F for 45 minutes.", "Rest 15 minutes before cutting.",
    ]], seed=15)  # fmt: skip
    path("Home/wifi-for-guests.txt").write_text(
        "Guest wifi\nNetwork: Rivera-Guest\nPassword: sunflower-river-42\nThe router is in the hallway "
        "closet if it needs a restart.\n"
    )
    ods(path("Home/vegetable-garden-plan.ods"), {
        "Beds": [["Bed", "Spring", "Summer"], ["Bed 1", "Lettuce, radishes", "Tomatoes, basil"],
                 ["Bed 2", "Peas", "Zucchini"], ["Bed 3", "Spinach", "Peppers, cucumbers"]],
    })  # fmt: skip
    xlsx(path("Home/car-maintenance-log.xlsx"), {
        "Honda Civic": [["Date", "Mileage", "Service", "Shop", "Cost"],
                        ["2023-11-04", "41200", "Oil change", "QuickLube Springfield", "54.99"],
                        ["2024-02-17", "44850", "New front brake pads", "Main Street Auto", "289.00"],
                        ["2024-05-30", "47900", "Oil change and tire rotation", "QuickLube Springfield", "79.98"]],
    })  # fmt: skip

    # Personal papers, and two other languages.
    scanned_pdf(path("Personal/birth-certificate-scan.pdf"), [[
        "**CERTIFICATE OF LIVE BIRTH", "Name of child: Maya Elena Rivera",
        "Date of birth: April 12, 2016   Time: 06:42", "Place of birth: Springfield General Hospital",
        "Mother: Alex Rivera   Father: Sam Rivera", "Registrar: County of Springfield",
    ]], seed=16)  # fmt: skip
    pdf(path("Personal/marriage-certificate.pdf"), [
        "CERTIFICATE OF MARRIAGE\n\nThis certifies that Sam Rivera and Alex Morgan were united in marriage "
        "on June 18, 2014 at the Riverside Botanical Garden, Springfield, by Judge Helen Brooks. "
        "Witnesses: Jordan Lee and Priya Shah."
    ])  # fmt: skip
    pdf(path("Personal/contrat-de-location-montreal.pdf"), [
        "BAIL DE LOGEMENT\n\nEntre le locateur, Gestion Immobilière Plateau, et le locataire, Sam Rivera, "
        "pour le logement situé au rue des Érables, appartement 3, Montréal.\n\nDurée du bail : du "
        "1er juillet 2017 au 30 juin 2018. Loyer mensuel : 1 150 $, payable le premier jour de chaque "
        "mois. Le chauffage et l'eau chaude sont inclus. Les animaux ne sont pas permis."
    ])  # fmt: skip
    path("Personal/facture-electricite-2018.txt").write_text(
        "Facture d'électricité - Énergie Laurentienne\nClient : Sam Rivera\nAdresse : rue des Érables, app. 3, "
        "Montréal\nPériode : 1er janvier au 28 février 2018\nConsommation : 1 240 kWh\nMontant à payer : "
        "96,45 $ avant le 20 mars 2018\n"
    )

    # An e-book.
    epub(path("Books/the-art-of-saving.epub"), "The Art of Saving", "Lena Brooks", [
        ("Pay yourself first", "Move money into savings the day your paycheck arrives, before you spend "
         "anything. Even 5% adds up.\n\nAutomatic transfers remove the willpower problem."),
        ("The emergency fund", "Aim for three to six months of expenses in a separate account. It turns "
         "a crisis into an inconvenience."),
        ("The 50/30/20 budget", "Half of your income for needs, 30% for wants and 20% for savings and "
         "paying off debt."),
    ])  # fmt: skip

    # Screenshots.
    screenshot("Cedar Credit Union", [
        "Payment sent", "$1,850.00 to Lakeside Property Mgmt", "From checking ****4471",
        "June 1, 2024 at 7:02 AM", "Reference 88231-RENT",
    ], accent=(46, 125, 50)).save(path("Screenshots/Screenshot_2024-06-01_bank.png"))  # fmt: skip
    screenshot("Alex", [
        "Dinner Saturday at 7?", "Yes! Where is it again?", "Luigi's Trattoria, 22 Harbor Rd",
        "Parking behind the building", "See you there 🍝",
    ], accent=(88, 86, 214)).save(path("Screenshots/Screenshot_2024-06-20_chat.png"))  # fmt: skip
    screenshot("City of Springfield", [
        "PARKING CITATION", "Violation: expired meter", "Fine: $45.00",
        "Location: 300 block of Main St", "Pay within 21 days to avoid a late fee",
    ], accent=(120, 120, 120)).save(path("Screenshots/IMG_2290.png"))  # fmt: skip

    # Notes and distractors.
    path("Notes/todo.txt").write_text(
        "- renew car registration before July 31\n- buy birthday present for Jordan\n- fix the squeaky "
        "bathroom door\n- book Maya's swimming lessons\n"
    )
    path("Notes/gift-ideas.md").write_text(
        "# Gift ideas\n- Alex: pottery class, noise-cancelling headphones\n- Jordan: board game "
        "(Wingspan), hot sauce set\n- Mom: framed family photo, gardening gloves\n"
    )
    path("Notes/books-to-read.md").write_text(
        "# Books to read\n- Project Hail Mary\n- The Psychology of Money\n- Braiding Sweetgrass\n"
    )
    path("Notes/saved-article-electric-cars.html").write_text(
        "<html><head><title>Are electric cars cheaper to own? - Daily Ledger</title></head><body>"
        "<article><h1>Are electric cars really cheaper to own?</h1><p>Over five years, lower fuel and "
        "maintenance costs often make up for a higher purchase price, especially with home charging."
        "</p></article><footer>Subscribe for more</footer></body></html>"
    )
    pdf(path("Documents/hoa-community-rules.pdf"), [
        "MAPLE STREET HOMEOWNERS ASSOCIATION - COMMUNITY RULES\n\nQuiet hours are 10 PM to 7 AM. Trash "
        "bins must be out of sight from the street except on collection day (Thursday). Fences may not "
        "be taller than six feet. Exterior paint colours must be approved by the board."
    ])  # fmt: skip
    pdf(path("Documents/software-terms-of-service.pdf"), [
        "TERMS OF SERVICE\n\nBy using the service you agree to these terms. The service is provided as is "
        "without warranty of any kind. We may update these terms at any time. Your data is processed "
        "according to our privacy policy. Disputes are resolved by binding arbitration."
    ])  # fmt: skip

    # Voice memos and narrated videos.
    if has_flite():
        speech(path("Voice/voice-memo-001.m4a"),
               "Note to self. Call the dentist on Tuesday to move my cleaning appointment, and pick up "
               "the dry cleaning on the way home.")  # fmt: skip
        speech(path("Voice/voice-memo-002.m4a"),
               "Ideas for Maya's birthday party. Rent a bounce house, order the chocolate cake from "
               "Rosie's bakery, and invite the whole class.")  # fmt: skip
        speech(path("Voice/budget-meeting-recording.m4a"),
               "Okay, let's start. In this meeting we agreed to cut the marketing budget by ten percent "
               "next quarter and to move the savings into hiring two more drivers.")  # fmt: skip
        speech(path("Videos/quarterly-update.mp4"),
               "Welcome to the quarterly update. Revenue grew twelve percent this quarter, and late "
               "deliveries fell to three percent.",
               picture=slide("Quarterly update", ["Revenue +12%", "Late deliveries 3.1%"]))  # fmt: skip
        speech(path("Videos/how-to-reset-the-router.mp4"),
               "If the internet stops working, unplug the router, wait thirty seconds, and plug it back "
               "in. Wait two minutes for the lights to turn green.",
               picture=slide("Resetting the router", ["Unplug", "Wait 30 seconds", "Plug back in"]))  # fmt: skip
    else:
        skipped.append("voice memos and videos (ffmpeg without flite)")

    # Plausible dates, so the files do not all look made a second ago.
    for i, rel in enumerate(sorted(made)):
        stamp = datetime(2024, 1, 1).timestamp() + rnd.randrange(0, 200) * 86400 + i
        os.utime(out / rel, (stamp, stamp))
    for s in skipped:
        print(f"warning: skipped {s}", file=sys.stderr)
    return made


# Searches of the hand-made core that now have more right answers among the generated
# files (another May statement, the vegetable lasagna, the sticky note with the wifi).
WIDEN = {
    "my resume": ["Career/Old resumes/"],
    "W-2 form": ["Taxes/w2-"],
    "Maya's report card": ["Kids/report-card-"],
    "flight booking to Lisbon": ["Travel/lisbon/flight-confirmation.pdf"],
    "where are we staying in Lisbon": ["Travel/lisbon/hotel-booking.html"],
    "blood test results": ["Health/Labs/"],
    "bank statement for May": ["Finance/Statements/cedar-checking-2023-05.pdf", "Finance/Statements/cedar-checking-2024-05.pdf"],
    "electric bill from March": ["Bills/Electric/metro-power-2023-03.pdf", "Bills/Electric/metro-power-2024-03.pdf"],
    "latest pay stub": ["Finance/Paystubs/brightline-paystub-2024-12-27.pdf"],
    "lasagna recipe": ["Recipes/vegetable-lasagna"],
    "packing list": ["Pictures/Notes/IMG_3106.jpg"],
    "dentist appointment": ["Pictures/Notes/IMG_3301.jpg"],
    "guest wifi password": ["Pictures/Notes/IMG_3300.jpg"],
    "grocery receipt": ["@FRESHWAY GROCERY"],
    "doctor stuff": ["Health/"],
}  # fmt: skip


def corpus(out: Path, cache: Path) -> tuple[list[str], list[dict], list[str]]:
    """Every file and search of the benchmark. Returns (files, searches, problems)."""
    import json

    import families
    import handwriting
    import media_files

    core = build(out)
    o = families.Out(out, files=list(core))
    for family in families.ALL:
        family(o)
    families.handwritten(o, handwriting.fonts(cache / "fonts"))
    problems = media_files.media(o, cache / "media")
    base = json.loads((Path(__file__).parent / "queries.json").read_text())["queries"]
    receipt_stores = families.receipt_stores(o)
    for item in base:
        for prefix in WIDEN.get(item["q"], []):
            if prefix.startswith("@"):
                item["want"] += receipt_stores.get(prefix[1:], [])
            else:
                item["want"] += [
                    f for f in o.files if f.startswith(prefix) and f not in item["want"]
                ]
    media_files.hard_queries(o, base + o.queries)
    return o.files, base + o.queries, problems


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_corpus.py OUT_DIR")
    out = Path(sys.argv[1])
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    files, searches, problems = corpus(out, Path.home() / ".cache" / "filesonar-benchmark")
    for p in problems:
        print(f"warning: {p}", file=sys.stderr)
    print(f"wrote {len(files)} files and {len(searches)} searches to {out}")


if __name__ == "__main__":
    main()
