# Arabic Token Tax Router

**Cost-efficient routing for Arabic customer service**
Future AI Builders Hackathon (HUMAIN) · Challenge 2: Deliver AI with less compute

A routing design for Arabic customer service: **GPT-4o decides which department handles each message, and the cheaper GPT-4o-mini writes the reply.** On 3,042 held-out Saudi-dialect banking messages, it matched GPT-4o's routing accuracy (82.0%) at **66% lower cost per correct answer**, with similar speed, and it held up across four Arabic dialects.

---

## The problem: the Arabic token tax

Arabic text breaks into more tokens than English, so every Arabic message costs more to process. For Saudi banks handling thousands of customer messages a day, running everything on one large model adds up quickly.

## The idea

Each customer message involves two different jobs:

| Job | What it needs | Model |
|---|---|---|
| **Routing:** which department should handle this? | Short, high-stakes decision | **GPT-4o** |
| **Reply:** write the Arabic answer | Long output, where most tokens are spent | **GPT-4o-mini** |

The expensive model only makes the short decision. The cheap model writes the long reply, grounded in the relevant department's policy paragraph.

```
Customer message (Arabic)
        │
        ▼
 GPT-4o ─► department (1 of 8)
        │
        ▼
 GPT-4o-mini + that department's policy ─► Arabic reply
```

## Data

- **Dataset:** [ArBanking77](https://github.com/SinaLab/ArBanking77) (ArabicNLP 2023), Arabic banking queries localized by native speakers from the English Banking77 dataset. Released for academic and commercial use.
- **Main dialect:** Saudi, 3,542 unique messages.
- **Departments:** the 77 intents were grouped into 8 departments: cards, security, transfers, top_ups, cash_atm, fees_exchange, card_payments, account.
- **Split:** dev (500) for design and prompt tuning; **test (3,042) kept untouched until the final run.**
- **Bank policy:** a short fictional policy (one paragraph per department) so replies are grounded rather than invented.

*Caveat:* the messages are localized from English, not raw customer chats.

## Results (3,042 unseen Saudi-dialect messages)

| Design | Routing accuracy | Cost per 1,000 correct answers | Avg seconds per message |
|---|---|---|---|
| Baseline: GPT-4o does everything | 82.0% | $2.16 | 1.66 |
| **Split design: GPT-4o routes, mini replies** | **82.0%** | **$0.73 (−66%)** | **1.74** |

### Robustness across dialects

Built and tuned on Saudi dialect only, then tested on 300 messages from each other dialect:

| Dialect | Accuracy | Cost per 1,000 correct | Seconds | Avg tokens per message |
|---|---|---|---|---|
| Saudi (test, 3,042) | 82.0% | $0.73 | 1.74 | 14.8 |
| MSA | 88.0% | $0.69 | 1.75 | 16.2 |
| Palestinian | 83.0% | $0.72 | 1.74 | 14.6 |
| Moroccan | 84.0% | $0.72 | 1.75 | 17.4 |

Accuracy, cost and speed held steady. Moroccan messages need about 18% more tokens than Saudi ones for the same questions, but that barely changes cost, because most of the cost is in the instructions and the reply, not the customer's message.

### Reply quality (blind judged)

Replies from both designs were judged blind, in random order, against the same policy paragraph, by an independent third-party model (Claude Sonnet).

| Measure | GPT-4o replies | GPT-4o-mini replies |
|---|---|---|
| Head-to-head preference | 36 | 36 (27 ties) |
| Stayed fully within policy | 88% | 74% |

**Honest trade-off:** customers would find the replies equally good, but mini fills gaps in the short synthetic policy with plausible invented details more often. A fuller real-world policy and a check-before-sending step (below) are the planned fixes.

## What didn't work: the confidence cascade

The first design let GPT-4o-mini route every message and escalate to GPT-4o only when unsure. It was tested and dropped: **mini is overconfident.** Half of its wrong answers scored above 99.99% confidence, so no confidence cutoff could separate right from wrong. That evidence is why the routing decision stays with GPT-4o.

## Repository contents

| File | What it is |
|---|---|
| `[ArabicTextTax].ipynb` | The full pipeline, step by step, with a text cell explaining each step |
| `big_test.csv` | Baseline results on the test set |
| `split_design_test.csv` | Final design results on the test set |
| `judge_test.csv` | Blind reply-quality judgments |
| `dialect_MSA.csv`, `dialect_Palestinian.csv`, `dialect_Moroccan.csv` | Dialect robustness results |
| `[other result files]` | Dev runs and the cascade experiment |

## How to run

1. Open the notebook in **Google Colab**.
2. Add your API keys in **Colab Secrets** (key icon in the left panel):
   - `OPENAI_API_KEY` for GPT-4o and GPT-4o-mini
   - `ANTHROPIC_API_KEY` for the reply judge
3. **To reproduce the results for free:** upload the result `.csv` files to the Colab file panel first. Every paid step checks whether its results file already exists and loads it instead of calling the API.
4. Run all cells from top to bottom. The notebook clones ArBanking77 automatically.


**Technologies:** Python, Google Colab, pandas, OpenAI API (GPT-4o, GPT-4o-mini), Anthropic API (judge), tiktoken.

## Next steps

- **Check before sending:** if mini's reply contains anything not in the policy, fall back to GPT-4o's reply ("verification where it pays").
- A fuller, more realistic bank policy to close the reply-quality gap.
- Native, untranslated customer messages and more domains beyond banking.

## Credits

ArBanking77: Jarrar et al., *ArBanking77: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic*, ArabicNLP 2023. [Paper](https://aclanthology.org/2023.arabicnlp-1.22)
