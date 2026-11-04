# Context for Claude: extract TwiBot-22 tweets in Google Colab

Written 4 Nov 2026, 16:45, to brief a separate Claude chat. For the whole project picture, read `00_handoff_for_claude.md` and `../reports/twibot22_plan.md`.

## Who and what

- **Project:** COMP8851 (Macquarie University). LLM + graph methods for bot/fraud detection under heterophily. Supervisor: Dr. Venus Haghighi.
- **Builder:** Prajan Bhattarai.
- **Deadline:** experiments done by **Sat 7 Nov 2026** night; Sun 8 Nov is for writing the report.
- **How the builder likes to work:**
  - Plain language, short answers.
  - Question every claim; mark anything from memory as unverified.
  - Before any job longer than ~30 min, say in one line what result it gives and why the report needs it.


## Steps for the user

1. Open the TwiBot-22 Drive folder link and choose **"Add shortcut to Drive"**. The folder then shows under My Drive, e.g. `/content/drive/MyDrive/TwiBot-22/`. Check the exact path after mounting.
2. In Colab, mount Drive: `from google.colab import drive; drive.mount('/content/drive')`.
3. Run the extraction notebook (below).

## The task

**Goal:** for each of the 1,000,000 TwiBot-22 users, keep their **20 most recent tweets** by `created_at`.

**Output:** one Parquet file, saved to Drive, e.g. `MyDrive/twibot22_out/tweets_recent20.parquet`.

Columns, the same as the local script `my_build/twibot22_tweets.py`:

| Column | Meaning |
|---|---|
| `user_id` | String, `"u" + author_id` (matches the ids in `label.csv`, e.g. `u1217628182611927040`) |
| `tweet_id` | Tweet id |
| `created_at` | String, e.g. `"2022-02-04 20:56:20+00:00"` |
| `lang` | Language code |
| `source` | Client name, with HTML tags stripped |
| `is_retweet` | `text` starts with `"RT @"`, or `referenced_tweets` has type `retweeted` |
| `is_reply` | `in_reply_to_user_id` is not null, or `referenced_tweets` has type `replied_to` |
| `text` | Tweet text |

Also save a small summary JSON:
- tweets read per file;
- users with ≥1 tweet;
- the distribution of tweets per user.

## Facts about the files (checked on partial downloads)

- Each file is **one JSON array** of tweet objects: `[{...}, {...}, ...]`.
- Keys are in **alphabetical order**, and every object starts with `{"attachments": ` and then `"author_id": <int>,`.
- **Example start:**
  `[{"attachments": null, "author_id": 16989141, "context_annotations": null, "conversation_id": 1493705209666805766, "created_at": "2022-02-15 21:54:08+00:00", "entities": {...}, ...`
- **`author_id` is a JSON integer.** User ids elsewhere in the dataset are strings with a `u` prefix.
- **Do not `json.load` a whole file.** At 12 GB it won't fit in Colab's ~12 GB RAM. Stream it instead:
  - **The local method:** split the text stream on the marker `{"attachments": `, read `author_id` with the regex `"author_id": (\d+),` on the first 200 characters, then `json.loads` each object after stripping a trailing `,` or `]`.
  - **Alternative:** `ijson`.
- **Size, from the TwiBot-22 paper:** about 88M tweets. That figure isn't checked against the files.
- **Our own checks:**
  - 93.4% of users have ≥1 collected tweet: 82.4% of bots, 95.2% of humans.
  - Median 53 tweets among users with any.
  - These come from the `post` edges in `edge.csv`, not from the tweet files. Re-check them on the real files.
- **The order of users within and across files is unknown.** A user's tweets may be spread over several files, so "most recent 20" must be decided after reading **all 9 files**, not per file.

## Memory plan (Colab free tier: ~12 GB RAM, sessions up to ~12 h, can disconnect)

These limits are from memory; check them.

1. **Per file:** stream the tweets and keep the slim columns above.
   - For each user, keep only that file's 20 most recent tweets, e.g. collect per chunk, then `groupby(user_id)` sort by date and `head(20)`.
   - Write `part_<i>.parquet` to Drive.
   - Record that file as done in a small JSON on Drive, so a disconnected session resumes where it stopped.
2. **After all 9:** concatenate the parts. Use DuckDB or pandas, reading only the columns needed, in chunks by `user_id` if memory is tight.
   - Take the top 20 by `created_at` per user and write the final file.
3. **Keep Drive storage in mind:** the free account has 15 GB. Delete the part files after the final file is checked.

## Rules

- **Licence:** TwiBot-22 is **CC BY-NC-ND 4.0**. Use it privately only.
  - Never make anything public: no public Kaggle/HF dataset, no git commit of data.
  - The Colab notebook can be shared; the data must not be.
- **Leave the user's original files alone:** the extraction only reads them.
- **Before the long run,** test on a slice, e.g. the first ~50,000 tweets of `tweet_0.json`:
  - check that `author_id` parses, the output columns, and speed per million tweets;
  - estimate the total time.

## What happens with the output (on the Mac, by the main project Claude)

1. The output becomes each user's text: **bio + 20 most recent tweets**, the same recipe as our TwiBot-20 work.
2. That text is embedded with MiniLM and multilingual-E5, either on the Mac GPU or on a Colab T4 if that's faster.
3. These embeddings are the text features for all TwiBot-22 models: MLP, XGB-Graph, DEC-lite trees, GNNs and the LLM sample.
