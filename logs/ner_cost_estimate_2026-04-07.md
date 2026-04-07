# NER Cost Estimate — 2026-04-07

## Sample data (from compare run)

| Model                        | Acts | Input tokens | Output tokens | Thinking tokens | EUR total |
|------------------------------|------|-------------|---------------|-----------------|-----------|
| gemini-3-flash-preview       | 20   | 24,000      | 4,700         | 0               | €0.0220   |
| gemini-3.1-flash-lite-preview | 5    | 5,800       | 895           | 632             | €0.0032   |

## Per-act cost

| Model                        | Input tok/act | Output tok/act | Thinking tok/act | EUR/act  |
|------------------------------|---------------|----------------|------------------|----------|
| gemini-3-flash-preview       | 1,200         | 235            | 0                | €0.00110 |
| gemini-3.1-flash-lite-preview | 1,160         | 179            | 126              | €0.00064 |

## Full dataset estimate (9,700 acts)

| Model                        | EUR total  | Ratio vs Flash |
|------------------------------|------------|----------------|
| gemini-3-flash-preview       | **€10.67** | 1.0×           |
| gemini-3.1-flash-lite-preview | **€6.21**  | 0.58×          |

Flash-lite is ~42% cheaper. Both are low enough that cost is not a deciding factor.

## Notes
- Flash-lite sample (n=5) is small; estimate less reliable than flash (n=20).
- Flash-lite thinking tokens add ~€0.0008 per 5 acts; at scale ~€1.55 extra vs zero thinking.
- Prices based on EUR rates active on 2026-04-07.
