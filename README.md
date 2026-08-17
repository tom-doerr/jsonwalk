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

```console
$ jsonwalk city_name is_in_europe -k 6
  #  value            P(value)   P(v&T)  D(T-F) P(true)  mass tok paths pre
  1  New York          0.05515  0.01483   -1.00    0.27 0.994   3     6
  2  Paris             0.04304  0.03667   +1.75    0.85 0.995   2     9
  3  London            0.03343  0.02793   +1.63    0.84 0.995   2     6
  4  Los Angeles       0.03265  0.00800   -1.13    0.25 0.994   3     5
  5  Berlin            0.01789  0.01524   +1.75    0.85 0.995   2     7
  6  Chicago           0.01779  0.00724   -0.38    0.41 0.994   2     7

266 distinct values seen, covering 43.1% of the probability mass; 145 expansions
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

The same value is reachable by several token sequences, and they are summed
(`logsumexp`) rather than ranked against each other: they are one answer, not
competing ones. It is worth being precise about what those paths actually are,
because it is not what you would guess. Dumping them for `Iron`:

```
0.035110 (97.6%)  ['Iron', '",']
0.000442 ( 1.2%)  ['Iron', '"},']
0.000415 ( 1.2%)  ['Iron', '"}']
0.000013 ( 0.0%)  ['Iron', '"']
0.000002 ( 0.0%)  ['Iron', '","']
0.000002 ( 0.0%)  ['Iron', '"],']
```

The body is tokenized identically every time. What varies is the **closing
token** — `",` `"}` `"},` `"` are each single tokens here, so the model shuts
the string and emits the following punctuation in one step, several different
ways. Genuine re-spellings of the body happen but are rare and low-probability,
because they mean deviating from the greedy BPE merge.

So merging is required for a correct `P(value)` — without it `Iron` is
understated by ~2.4% — but on this model the top path carries ~97% of a value's
mass, so it seldom reorders the table. It matters most for values whose
segmentation is genuinely ambiguous. Highlight a row in the TUI and the bottom
pane shows its paths.

## The columns

| Column | Meaning |
| --- | --- |
| `P(value)` | Probability of the whole string, merged over tokenizations. |
| `P(v&T)` | `P(value)` × `P(true)` — likely **and** true. See sorting below. |
| `D(T-F)` | `log P(true…) − log P(false…)`. A log-odds: `+2.3` means the model is ~10× more willing to write `true`. **This is the judgement.** |
| `P(true)` | `sigmoid(D)` — the verdict as a probability, *given* the model writes a boolean at all. |
| `mass` | `P(true…) + P(false…)` in absolute terms. **This is the sanity check.** |
| `tok` / `paths` | Tokens in the best spelling; token sequences merged into the row. |
| `pre` | `*` — the value appears verbatim in your preamble. |

`true` and `True` are pooled, as are `false` and `False`; a spelling
preference is not an opinion.

`mass` is the column to look at first. If it is low, the model did not intend
to write a boolean in that slot at all, and the delta is a ratio between two
things it never wanted to say — the fix is the preamble, not more search. With
no preamble the model's favourite continuation in the boolean slot is a
*digit*: it reads `good_sounding_name` as numeric. The CLI and TUI both warn
when it drops below 0.5.

## Sorting

`--sort` on the CLI, `ctrl+s` cycles in the TUI.

| Mode | Ranks by | Answers |
| --- | --- | --- |
| `signal` (default) | `P(value)` × `D(T-F)` | What is likely **and** true? |
| `joint` | `P(value)` × `P(true)` | Probability of the whole object. |
| `value` | `P(value)` | What would it write here? |
| `delta` | `D(T-F)` | What does it most believe, however unlikely? |

Scored against hand-labelled ground truth over the whole 100-candidate pool.
`precision@20` counts labelled rows in the displayed list; `AUC` is the
fraction of (true, false) pairs the full ordering gets the right way round:

| Mode | `element`/`is_metal` | `city_name`/`is_in_europe` |
| --- | --- | --- |
| `signal` | **1.00 / 1.000** | **1.00 / 1.000** |
| `joint` | 0.94 / 0.966 | 0.41 / 0.597 |
| `value` | 0.68 / 0.664 | 0.35 / 0.364 |
| `delta` | 1.00 / 1.000 | 1.00 / 1.000 |

Sorting by likelihood alone interleaves metals and nonmetals by nothing but
word frequency; `signal` puts **all the metals above all the nonmetals**:

```console
$ jsonwalk element is_metal -k 14
  4  Copper       +0.0660  0.02933   +2.25    0.90
  2  Iron         +0.0583  0.03599   +1.62    0.84
  5  Sodium       +0.0444  0.01974   +2.25    0.90
  ...
  7  Carbon       -0.0641  0.01710   -3.75    0.02
  1  Hydrogen     -0.1122  0.06001   -1.87    0.13  <- most likely value, and rejected
  3  Helium       -0.1154  0.03297   -3.50    0.03
```

### Why the signed delta, and not `P(true)`

`P(true)` is `sigmoid(D)`, and it **saturates**: +1.4 and +2.3 nats are only
0.80 versus 0.90, a factor of 1.1, while `P(value)` varies by 10× across a
list. So in `joint` the likelihood swamps the verdict — which is how "cities
in Europe" ends up with New York, Los Angeles, Chicago and Tokyo in its top
20, at precision 0.41.

Multiplying by the raw delta instead keeps its **sign**, and that changes the
structure of the ranking: every positive score beats every negative one, so
`P(value)` orders *within* a verdict and never across it. A value the model
rejects cannot float up on popularity — it sinks in proportion to how common
it is, which puts the most likely false value at the very bottom.

`delta` alone separates just as perfectly, being sign-carrying too. What
`P(value)` buys is the ordering *inside* the true group, regularising away the
rare and the malformed:

```
signal  Paris, London, Berlin, Rome, Barcelona
delta   Lyon, Bordeaux, Rome, Dublin, Hamburg     <- equally European, less useful
```

It is a ranking heuristic, not a probability: multiplying a probability by a
log-odds has no distributional meaning. `joint` is the one with a clean
interpretation, and it is kept for when that matters.

The `#` column keeps the original likelihood rank, so you can see how far each
row moved.

### Sorting happens before `k` is chosen

`k` is how many rows you *see*. The walk and the boolean scorer both work on a
pool — `k × --pool-factor`, 5 by default — and the k are picked from it
**after** sorting. Selecting first would make every mode a reshuffle of the
most-likely values, so a value ranked 14th by likelihood but first by verdict
could never appear.

That is not hypothetical. On `element` / `is_metal` with `k=8`:

| | Result |
| --- | --- |
| `--pool-factor 1` (select, then sort) | 5 metals, padded with Hydrogen, Helium, Carbon |
| default pool | **8 metals** — Silver (#9), Lithium (#10) and Aluminium (#14) displace them |

Cost, model already loaded: 0.97 s at factor 1, **2.12 s at factor 5**. Factor
10 jumps to 9.4 s, because resolving that far down a flat tail costs the
search 1297 expansions instead of 241. Switching sort mode in the TUI re-picks
from the pool with no model call at all.

## The preamble matters more than anything else

This is a base model completing a document, so the preamble decides both which
values appear and whether the boolean slot means anything. Measured on
`Qwen3.5-0.8B-Base`:

| `--preamble-style` | `mass` | What you get |
| --- | --- | --- |
| none at all | 0.04 – 0.20 | the boolean slot fills with a *digit* |
| `comment` | 0.70 – 0.80 | right values, weak verdicts (New York only +0.07) |
| `json-schema` | 0.89 – 0.96 | **field names as values** — see below |
| examples reusing the queried field name | 0.997 | asking `city_name` returns **Stripe, Google, Amazon** |
| `examples` from unrelated domains (default) | 0.82 – 0.99 | right values, sharpest verdicts |

### Would a JSON Schema work as the preamble?

It is the obvious thing to try, and measured on this model it is worse. Ship
it with `--preamble-style json-schema` and see for yourself; here is why it
loses.

A base model continues the pattern it was shown, and a JSON Schema is a
pattern of *type declarations*, not of *records*. What it does show the model
is your field names, which it then writes as values:

```
$ jsonwalk startup_name good_sounding_name --preamble-style json-schema
   Apple | My Startup | Good Sounding | Hello World | good_sounding | good_sounding_na
```

`element` degrades the same way (`metal, steel, a, element, 1, foo`). The
`pre` column catches it automatically — three of eight rows were flagged as
appearing in the preamble, which is how the failure was spotted.

A schema plus one worked example fixes the field-name leak but hands the whole
domain to that example: with a film example, `element` returns *Casablanca,
character, film, actor, director*. Two examples from two different domains —
the default — dilute the anchor enough to keep both the schema and the values.

The trap in row three is worth spelling out: worked examples buy the highest
`mass` of all, but if they use the field you are asking about, they hijack it
— ask for `city_name` with startup examples and you get startups. The default
therefore uses examples about films and chemical elements, which teach the
*shape* of the object without supplying its subject. A third example made both
the separation and the mass worse.

**Examples still get echoed back as candidates, and the `pre` column marks
it.** The default preamble names Helium, so asking for `element` puts Helium
at rank 3 — delete that example line and Helium leaves the top 14 entirely.
But the same run then degrades to `water, person, a, element, Nile` (`Nile`
being an echo of the *other* example): the chemistry line is what makes
`element` produce elements at all. The echo is the price of the priming, so
the tool flags it rather than pretending otherwise.

`{field}` and `{bool_field}` in a preamble expand to the field names in use,
if you do want same-domain examples for a fixed task.

## Install

```bash
git clone git@github.com:tom-doerr/jsonwalk.git
cd jsonwalk
pipx install --system-site-packages --editable .
```

That puts a `jsonwalk` command on your `PATH`. Needs `torch`, `transformers`
and `textual`; the model (~1.7 GB) downloads from the Hub on first run.

**`--system-site-packages` is the load-bearing flag.** Without it pipx builds
a sealed venv and pulls its own `torch` — several GB, and on hardware with a
purpose-built wheel (a CUDA 13 / GB10 box, say) very likely the wrong one.
With it, the venv reuses the interpreter's existing `torch`, and pip installs
nothing but `jsonwalk` itself. `--editable` keeps the checkout as the source
of truth, so edits take effect with no reinstall.

Plain `pip install -e .` works too wherever the environment is not externally
managed (PEP 668). To run with no install at all:

```bash
PYTHONPATH=src python3 -m jsonwalk city_name is_in_europe
```

Uninstall with `pipx uninstall jsonwalk`.

## Use

Give it two field names and it prints a table:

```bash
jsonwalk city_name is_in_europe             # the common case
jsonwalk startup_name good_name -k 30       # more candidates
jsonwalk pet_name is_cute --objects         # one JSON object per line
jsonwalk pet_name is_cute --json | jq .     # everything, machine readable
jsonwalk                                    # no arguments: the TUI
```

`--objects` is the pipe-friendly form:

```console
$ jsonwalk pet_name is_cute -k 4 --objects
{"pet_name": "Rex", "is_cute": true}
{"pet_name": "Whiskers", "is_cute": true}
{"pet_name": "Buddy", "is_cute": true}
{"pet_name": "Golden Retriever", "is_cute": true}
```

`run` and `tui` are reserved as the first word (`jsonwalk run …` is the
explicit form of the default, `jsonwalk tui` opens the interface). A field
actually named `run` needs `jsonwalk run run <bool_field>`.

### The TUI

Run `jsonwalk` with no arguments. Three stacked inputs, then the results:

```
╭──────────────────────────────────────────────────────────────╮
│ string field  startup_name                                   │
│ bool field    good_sounding_name                             │
│ values (k)    20                                             │
╰──────────────────────────────────────────────────────────────╯
 Run   Sort   Preamble   Help
╰──────────────────────────────────────────────────────────────╯
 enter or ctrl+r to run   -   F1 explains every column
 #   value              P(value)   tok  paths  D(T-F)  P(true)
 1   Google              0.02139     2      4   +1.13     0.75
```

| Key | Action |
| --- | --- |
| `ctrl+r` / `enter` | Run |
| `ctrl+s` | Cycle sort: likelihood / likely-and-true / verdict |
| `ctrl+y` | Copy every row as JSON |
| `ctrl+b` / `F2` | Edit the preamble |
| `ctrl+g` / `F1` | Help: what `k` is, what every column means |
| `ctrl+q` | Quit |

Nothing requires a function key — `F1`/`F2` are aliases, and every action also
has a button on the main screen. The `ctrl` choices are constrained: `Input`
claims `ctrl+a/e/d/f/k/u/w/x/v` for line editing and `ctrl+p` is Textual's
command palette, so binding any of those would break typing in a field. A test
asserts none of them are used.

One input per row is deliberate. Laid out side by side they collapse to two
or three characters each on a narrow window, and the preamble is a whole
document rather than a field — it gets its own screen (`F2`) instead of
taking six rows off the table.

As a library:

```python
from jsonwalk import RunConfig, run
from jsonwalk.hf import HFLanguageModel

lm = HFLanguageModel()
result = run(lm, RunConfig(field="city_name", bool_field="is_in_europe"))
for row in result.rows:
    print(row.as_object(result.schema), row.bool_score.delta)
```

## `jsonwalk edit` — the same trick applied to editing text

Same three ideas, one more field:

```json
{"search": "in order to", "replace": "to", "improves_objective": true}
```

```bash
jsonwalk edit draft.md -o "Make it concise."
jsonwalk edit draft.md -o "Remove hedging." --sort delta
jsonwalk edit draft.md -o "Be concise." --iterations 3 --in-place
```

The point is the constraint. `search` is walked under `SubstringOf(document)`,
and because **substring-ness is prefix-closed**, that is a sound *incremental*
test: a branch dies the moment it leaves the document, rather than being
generated and rejected afterwards. On a two-line file that pruned **8425
branches in 328 expansions**.

So the usual failure of model-written search/replace — an anchor that is not
in the file — is not filtered out, it is **unreachable**. Anchors are also
required to be unique, so an edit always applies to exactly the intended spot.
`--min-anchor` sets a length floor, which matters because shorter strings are
always more probable.

`replace` is then walked conditioned on the chosen anchor, and each pair is
judged against your stated objective. The objective genuinely steers it — the
same document and the same anchors, in opposite directions:

| anchor | `-o "Make it concise"` | `-o "Make it formal and verbose"` |
| --- | --- | --- |
| `we are currently` | → `we are` | → `we are presently` |
| `we are` | → `we` | → `we are currently` |
| `optimization` | → `optimize` | → `workflow optimization` |

Nothing is written without `--in-place`, and you always get a unified diff.

**What a 0.8B judge is actually good for.** It finds real edits
(`"In the event that a build fails" → "if a build fails"`,
`"configuration file" → "config"`) but it is not a careful editor: over three
`--iterations` it also produced `"if a build fails" → "if build fails"`,
dropping an article, and it never touched the wordiest clause in the file.
Anchor choice is driven by `P(search)`, so it edits where the text is
*predictable*, not where it is *worst*. Point `--model` at something larger if
you want the judgement to carry weight — the constraint machinery is
model-independent.

## Known trade-offs

* **Few-shot examples are a bias you are choosing to accept.** They are what
  makes the boolean slot mean anything, and the model will echo them — the
  default keeps them off-topic so the echoes are obvious rather than
  plausible, and the `pre` column flags them. `--preamble-style comment` drops
  them entirely, at the cost of a much weaker boolean signal.
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

44 tests in under a second, with no GPU and no model download: the search runs
against a deterministic toy model with a hand-written probability table,
including a vocabulary where one string has two tokenizations, and is
cross-checked against exhaustive enumeration. The TUI is exercised headless.
One test asserts that importing the CLI does not import torch — that property
is what keeps `jsonwalk --help` at 40 ms, and it is easy to break by accident
with a stray top-level import.

## Licence

MIT.



