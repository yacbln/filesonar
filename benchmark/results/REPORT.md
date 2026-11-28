# filesonar benchmark (981 files, 420 searches)

filesonar 0.3.2 on Linux x86_64, x86_64, NVIDIA GeForce RTX 3090. Models: nomic-embed-text (text), qwen2.5vl:7b (pictures), whisper base (speech). Indexing took 8 min 12 s.

| | filename | keyword | meaning | filesonar |
|---|---|---|---|---|
| Right file first | 34% | 76% | 88% | 86% |
| Right file in top 3 | 40% | 87% | 93% | 94% |
| MRR | 0.39 | 0.82 | 0.91 | 0.90 |
| 95% range, top 3 | 36%–45% | 84%–90% | 91%–96% | 92%–96% |
| Time per search | 1.1 ms | 1.6 ms | 8.4 ms | 9.7 ms |

## Right file in the top 3, by kind of search

| Kind | Searches | filename | keyword | meaning | filesonar |
|---|---|---|---|---|---|
| describe | 144 | 71% | 96% | 99% | 99% |
| question | 75 | 39% | 89% | 96% | 100% |
| exact | 4 | 75% | 100% | 75% | 100% |
| visual | 73 | 16% | 89% | 96% | 95% |
| photo | 48 | 0% | 90% | 96% | 98% |
| spoken | 16 | 25% | 81% | 94% | 94% |
| sound | 12 | 8% | 0% | 0% | 0% |
| video | 6 | 0% | 83% | 100% | 100% |
| language | 3 | 67% | 100% | 100% | 100% |
| typo | 12 | 50% | 50% | 83% | 83% |
| vague | 7 | 29% | 71% | 71% | 86% |
| absent | 20 | 25% | 5% | 15% | 5% |

## filesonar by file type (right file in the top 3)

| Type | Searches | Top 1 | Top 3 |
|---|---|---|---|
| .docx | 26 | 96% | 100% |
| .epub | 6 | 100% | 100% |
| .html | 13 | 100% | 100% |
| .jpg | 89 | 92% | 99% |
| .m4a | 11 | 73% | 91% |
| .md | 20 | 85% | 100% |
| .mp3 | 12 | 0% | 0% |
| .mp4 | 10 | 100% | 100% |
| .pdf | 138 | 85% | 96% |
| .png | 19 | 95% | 95% |
| .pptx | 5 | 100% | 100% |
| .txt | 19 | 74% | 100% |
| .xlsx | 8 | 88% | 100% |
| mixed | 18 | 78% | 94% |
