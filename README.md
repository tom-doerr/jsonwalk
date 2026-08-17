# jsonwalk

A TUI that walks a base language model's completion tree to enumerate the
likely values of a JSON string field, then asks the model how *true* a
statement about each one is by comparing the logprob of `true` against
`false` in a companion boolean field.

The object under construction is always this shape:

```json
{"startup_name": "Stripe", "good_sounding_name": true}
```

You name the string field, jsonwalk lists its most likely values in order.
You name the boolean field, jsonwalk scores every value with it. Ask for
`startup_name` + `good_sounding_name` and you get a ranked list of plausible
startup names, each with the model's opinion of whether it sounds good.

```
$ jsonwalk run city_name is_in_europe -k 6

  #  value                             P(value) tok paths  D(T-F) P(true) bool_mass
-----------------------------------------------------------------------------------
  1  New York                           0.05515   3     6   -1.00    0.27     0.994
  2  Paris                              0.04304   2     9   +1.75    0.85     0.995
  3  London                             0.03343   2     6   +1.63    0.84     0.995
  4  Los Angeles                        0.03265   3     5   -1.13    0.25     0.994
  5  Berlin                             0.01789   2     7   +1.75    0.85     0.995
  6  Chicago                            0.01779   2     7   -0.38    0.41     0.994

266 distinct values seen, covering 43.1% of the probability mass; 145 expansions; top-k provably complete
```

Nobody told it which cities are in Europe. The sign of `D(T-F)` is the model's
answer, and on a question with a checkable ground truth it gets all six right.

Default model: `Qwen/Qwen3.5-0.8B-Base`. A *base* model matters here — the
whole method depends on the model continuing a JSON document rather than
answering a chat turn.

## Why a tree walk

Values are strings, and strings do not have a fixed token length. `Uber` is
one token; `Amazon Web Services` is four. Two approaches that look obvious
both get this wrong:

* **Sampling** gives you frequencies, and needs a great many draws before the
  tail is ranked correctly.
* **Beam search** fixes the width per *step*, which quietly favours whichever
  values happen to use fewer tokens.

jsonwalk keeps a priority queue of partial values ordered by cumulative
logprob, so a one-token value and an eleven-token value compete on the same
axis: the joint probability of the whole string. Because every extra token
can only lower a path's logprob, the best score left on the frontier is an
upper bound on everything unexplored — once it falls below the k-th best
finished value, the top-k is provably complete.

A value is finished at the **first unescaped `"`**, which may arrive in the
middle of a token: `",` and `"}` are single tokens in this vocabulary, so the
walker scans inside each candidate token rather than waiting for a lone quote.
`\"` does not count.

### Merging tokenizations

The same string is often reachable by several different token sequences.
In the run above `Paris` arrived **9 ways** and `Berlin` **7** (see the
`paths` column). Those are not competing answers — they are one answer spelled
differently, so their probabilities are summed (`logsumexp`), not ranked
against each other.

This changes results, not just bookkeeping: a value whose mass is split across
several tokenizations can lose to a rival on any single path and still win
once merged.

## The three numbers

| Column | Meaning |
| --- | --- |
| `P(value)` | Joint probability of the whole string, merged over tokenizations. |
| `D(T-F)` | `log P(true…) − log P(false…)`. A log-odds: `+2.3` means the model is ~10× more willing to write `true`. **This is the judgement.** |
| `P(true)` | `sigmoid(D)` — the verdict as a probability, *given* the model writes a boolean at all. |
| `bool_mass` | `P(true…) + P(false…)` in absolute terms. **This is the sanity check.** |

`true` and `True` are pooled, as are `false` and `False`; a spelling
preference is not an opinion.

`bool_mass` is the column to look at first. If it is low, the model did not
intend to write a boolean in that slot at all, and the delta is a ratio
between two things it never wanted to say — the fix is the preamble, not more
search. With no preamble the model's favourite continuation in the boolean
slot is a *digit*: it reads `good_sounding_name` as numeric. The CLI and TUI
both warn when `bool_mass` drops below 0.5.

## The preamble matters more than anything else

This is a base model completing a document, so the preamble decides both which
values appear and whether the boolean slot means anything. Measured on
`Qwen3.5-0.8B-Base`:

| Preamble | `bool_mass` | `city_name` returns |
| --- | --- | --- |
| none | 0.04 – 0.20 | — |
| schema comment (`--schema-only-preamble`) | 0.72 – 0.80 | cities, weak verdicts (New York only +0.07) |
| examples reusing the queried field name | 0.997 | **Stripe, Google, Amazon** |
| examples from unrelated domains (default) | 0.82 – 0.99 | cities, sharp verdicts |

The trap in row three is worth spelling out: worked examples buy the highest
`bool_mass` of all, but if they use the field you are asking about, they
hijack it — ask for `city_name` with startup examples and you get startups.
The default therefore uses examples about films and chemical elements, which
teach the *shape* of the object without supplying its subject. A third
example made both the separation and the mass worse.

`{field}` and `{bool_field}` in a preamble expand to the field names in use,
if you do want same-domain examples for a fixed task.

## Install

```bash
git clone git@github.com:tom-doerr/jsonwalk.git
cd jsonwalk
pip install -e ".[dev]"
```

Needs `torch`, `transformers` and `textual`. The model (~1.7 GB) downloads
from the Hub on first run.

If those three are already present system-wide (and `pip install` refuses
because the environment is externally managed, PEP 668), skip the install and
run straight from the checkout:

```bash
PYTHONPATH=src python3 -m jsonwalk          # TUI
PYTHONPATH=src python3 -m jsonwalk run      # headless
```

## Use

```bash
jsonwalk                              # the TUI
jsonwalk run                          # headless, default fields
jsonwalk run city_name is_in_europe -k 30
jsonwalk run product_name is_expensive --objects   # one JSON object per line
jsonwalk run startup_name good_name --json | jq '.rows[0]'
```

TUI keys: `ctrl+r` run, `ctrl+s` toggle sorting between value likelihood and
true/false delta, `ctrl+y` copy every row as JSON, `ctrl+q` quit. Edit the
preamble in the box at the top; `{field}` and `{bool_field}` expand to the
field names you typed.

As a library:

```python
from jsonwalk import RunConfig, run
from jsonwalk.hf import HFLanguageModel

lm = HFLanguageModel()
result = run(lm, RunConfig(field="city_name", bool_field="is_in_europe"))
for row in result.rows:
    print(row.as_object(result.schema), row.bool_score.delta)
```

## Known trade-offs

* **Few-shot examples are a bias you are choosing to accept.** They are what
  makes the boolean slot mean anything, and the model will echo them — the
  default keeps them off-topic so the echoes are obvious rather than
  plausible. `--schema-only-preamble` drops them entirely, at the cost of a
  much weaker boolean signal.
* **Completeness has a caveat.** The frontier bound is exact for single
  paths. Merging means a brand-new value whose many tokenizations *sum* above
  the cut-off could in principle be missed, so the search keeps going until
  remaining paths are below `stop_margin` (default 5%) of the k-th best.
* **Nothing is silently dropped.** Paths that hit the token cap, run into a
  raw control character, or emit EOS without closing the string are counted
  in `WalkStats` rather than discarded quietly. `found_mass` reports what
  fraction of the total probability the returned values actually cover — on a
  flat field that is a few percent, and it should be read as such.
* **A 0.8B base model is a weak judge.** Deltas here run ±2 nats, not ±20.
  It gets `is_in_europe` right across six cities; it will not adjudicate
  anything subtle. Point `--model` at something larger if you need it to.

## Implementation notes

Qwen3.5 is a hybrid: most layers are `linear_attention` with full attention
every fourth layer. Those keep a recurrent state rather than a forkable KV
cache, so the walker re-forwards whole prefixes in batches instead of
branching from a cached node — cheap at 0.8B, and exact.

The vocabulary is ~248k tokens, which makes one position of fp32 logits about
1 MB. Every forward therefore passes `logits_to_keep`, and sequences are
bucketed by length so nothing needs padding (`logits_to_keep` counts from the
end of the tensor, which is the wrong place for a right-padded short row).

The prompt for the boolean stops at the colon, with no trailing space — the
space belongs to the scored word. Otherwise the tokenizer merges that space
into ` true` when encoding the full text, the prompt stops being a token
prefix of it, and the measured probability quietly refers to a different
context. `score_bools` raises `TokenBoundaryError` if that ever happens.

## Tests

```bash
pytest
```

37 tests, no GPU and no model download required: the search runs against a
deterministic toy model with a hand-written probability table, including a
vocabulary where one string has two tokenizations, and is cross-checked
against exhaustive enumeration. The TUI is exercised headless.

## Licence

MIT.



