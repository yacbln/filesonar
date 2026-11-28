# filesonar benchmark (981 files, 420 searches)

filesonar 0.3.2 on Linux x86_64, x86_64, NVIDIA GeForce RTX 3090. Models: embeddinggemma (text), qwen2.5vl:7b (pictures), whisper base (speech), ast-audioset (sounds). Indexing took 10 min 11 s.

| | filename | keyword | meaning | filesonar |
|---|---|---|---|---|
| Right file first | 34% | 78% | 95% | 94% |
| Right file in top 3 | 40% | 90% | 96% | 98% |
| MRR | 0.39 | 0.84 | 0.96 | 0.96 |
| 95% range, top 3 | 36%–45% | 87%–93% | 94%–98% | 97%–100% |
| Time per search | 1.0 ms | 1.8 ms | 11.7 ms | 13.4 ms |
| Right file first, yet flagged unsure | 0.0% | 6.0% | 2.0% | 1.5% |

## Right file in the top 3, by kind of search

| Kind | Searches | filename | keyword | meaning | filesonar |
|---|---|---|---|---|---|
| describe | 144 | 71% | 96% | 100% | 99% |
| question | 75 | 39% | 88% | 97% | 99% |
| exact | 4 | 75% | 100% | 75% | 100% |
| visual | 73 | 16% | 89% | 100% | 100% |
| photo | 48 | 0% | 90% | 96% | 98% |
| spoken | 16 | 25% | 81% | 94% | 94% |
| sound | 12 | 8% | 92% | 75% | 100% |
| video | 6 | 0% | 83% | 100% | 100% |
| language | 3 | 67% | 100% | 100% | 100% |
| typo | 12 | 50% | 50% | 75% | 92% |
| vague | 7 | 29% | 71% | 57% | 86% |
| absent | 20 | 25% | 90% | 90% | 80% |

*absent*: searches for things that are not there. They count as right when nothing comes back or the search says nothing clearly matches (`confidence: low`).

## filesonar by file type (right file in the top 3)

| Type | Searches | Top 1 | Top 3 |
|---|---|---|---|
| .docx | 26 | 100% | 100% |
| .epub | 6 | 100% | 100% |
| .html | 13 | 100% | 100% |
| .jpg | 89 | 96% | 99% |
| .m4a | 11 | 91% | 91% |
| .md | 20 | 90% | 100% |
| .mp3 | 12 | 75% | 100% |
| .mp4 | 10 | 100% | 100% |
| .pdf | 138 | 92% | 99% |
| .png | 19 | 95% | 95% |
| .pptx | 5 | 100% | 100% |
| .txt | 19 | 95% | 100% |
| .xlsx | 8 | 100% | 100% |
| mixed | 18 | 83% | 94% |
