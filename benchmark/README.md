# filesonar benchmark

How well does filesonar find a file when you describe it? This benchmark measures that on
a realistic, reproducible set of about 1,000 personal files, with a few hundred searches
phrased the way people ask, and compares filesonar with simpler ways of searching.

```bash
python benchmark/run.py --report benchmark/results/REPORT.md
```

It needs what filesonar needs (Ollama with the text and vision models; speech recognition
for recordings, see `filesonar setup`) and about 15 minutes on a machine with a GPU. It
works in a temporary folder and never touches your own files or index.

## The files

One made-up family, the Riveras, with the things people really keep, including the
look-alikes that make finding the right one hard:

| | |
|---|---|
| Money | 24 monthly bank statements, 26 pay stubs, tax returns, W-2 scans and 1099s across years, a budget spreadsheet, a car loan |
| Bills | 24 electric, 12 water and 12 phone bills (one with roaming charges, one warning about a leak) |
| Receipts | 65 phone photos of receipts from 20 stores, slightly tilted, blurred and noisy |
| Home and car | insurance policies (car, home, renters), the lease and its renewal, 25 appliance manuals, car service invoices (some scanned), contracts and memberships |
| Work and career | 40 job postings, five résumé versions, cover letters, an offer letter, 30 meeting notes, slides and spreadsheets |
| Health and school | lab results by year, prescriptions, an eyeglass prescription (scanned), report cards (scanned), newsletters, permission slips, homework |
| Travel | flight confirmations, hotel bookings and itineraries for eight trips, a boarding pass screenshot |
| Notes and recipes | 150 notes (a few with the one fact you will need), 43 recipes as text, Markdown, Word, PDF and scans |
| Pictures of paper | handwritten notebook pages, whiteboards and sticky notes with known text (rendered with handwriting fonts and photo effects), plus real photos of whiteboards, handwritten letters, sticky notes, menus, posters, street signs and maps |
| Photos | about 200 real photos of what people photograph: pets, food, beaches, hikes, snow, landmarks, weddings, concerts, gardens, cars, airports |
| Screenshots | chats, bank alerts, delivery notices, tickets, error messages |
| Recordings and videos | voice memos and narrated videos (synthesized speech), real sounds (rain, birds, applause, music), and short videos made of a real photo and a real sound |
| Other languages | a French lease and electricity bill (an old apartment in Montreal) |

Every person, company, address and number is invented. Formats: PDF (text and scanned),
Word, Excel, PowerPoint, OpenDocument, EPUB, HTML, Markdown, text, CSV, JPEG, PNG, M4A,
MP3 and MP4.

### Where the real photos and sounds come from

`media.json` lists every downloaded photo and sound with its title, creator, license and
source page. They were found with [Openverse](https://openverse.org) and are **all CC0 or
public domain**. They are downloaded when the benchmark runs (about 25 MB, cached in
`~/.cache/filesonar-benchmark`) and checked against fixed checksums, so everyone measures
on exactly the same files. The images get neutral camera names (`IMG_5012.jpg`), so
they can only be found by what they show. Handwriting fonts (Caveat, Patrick Hand, Indie
Flower, Permanent Marker) come from Google Fonts under the SIL Open Font License or the
Apache License, also checked against checksums. `pick_media.py` is how the list was made. Search
results for a subject are loose (a "Times Square" search returned churches), so every
photo was then checked by eye and the ones that do not show their subject were removed.

## The searches

Each search lists the files that count as a right answer; a search like "my resume"
accepts any version of the résumé, "photos of our dog" any photo of a dog. Kinds:

| Kind | Example |
|---|---|
| describe | "the water bill from this past March" |
| question | "how much did I pay for the air fryer", "is Maya allergic to cats" |
| exact | "order number 512-8830114", "booking reference K7Q2ZP" |
| visual | "where's the receipt for the headlamp", "whiteboard with the Q3 roadmap" |
| photo | "sunset at the beach", "photos of whiteboards" |
| spoken / sound / video | "voice note about the book club pick", "recording of rain" |
| typo | "reciept for the dishwaser" |
| vague | "car stuff", "stuff from our trip to Japan" |
| language | "my old lease in Montreal", "contrat de location" |
| absent | "boat registration": nothing like it exists, so the right answer is no result |

Searches are written apart from the documents' own wording, the way a person remembers a
file rather than quoting it.

## What is compared

| Method | What it is |
|---|---|
| filename | words of the search matched against file names, what a file manager does |
| keyword | filesonar's keyword index alone (BM25 over the text it extracted) |
| meaning | filesonar's meaning search alone (embeddings) |
| filesonar | the hybrid filesonar uses: both, fused, with file-name matches counting extra |

All four search the same index, so the differences come from the search method alone. For
each search the benchmark checks where the first right file appears: first (top 1), in
the first three (what the agent shows you), and the mean reciprocal rank. Ranges are 95%
bootstrap intervals.

## What it does and does not show

- It is synthetic. Real collections are messier: more duplicates, more junk, more files
  per family. The photos, sounds and handwriting fonts are real; the documents are not.
- The searches were written by the people who built the files. They try not to reuse the
  documents' wording, but they are not a survey of real users.
- Recordings use synthesized speech (flite). It is robotic rather than clean: speech
  recognition gets about one word in five wrong, more than on a clear human voice.
- Absent searches show whether filesonar can say "nothing found". It returns its closest
  matches today, so it scores poorly there; this is listed as a known gap.

Contributions that make it harder or more realistic are welcome: more families,
better searches, other languages.

## Results

See [results/REPORT.md](results/REPORT.md) for the latest run.
