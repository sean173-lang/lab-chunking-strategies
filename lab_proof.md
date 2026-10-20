# Lab Proof — Chunking Strategies

## 1. Workflow and Implementation

The objective of this lab was to compare three text chunking strategies and evaluate how they affect document structure, chunk size, and information retrieval.

I worked with two different sources: a podcast about trustworthy AI and the European Commission's *Ethics Guidelines for Trustworthy AI*. These sources provided an opportunity to compare conversational language with a longer, structured policy document.

I used Python 3.11 in a Jupyter Notebook, together with `langchain-text-splitters`, `pypdf`, `openai`, `pandas`, and `matplotlib`.

First, I compressed the original 29 MB podcast recording into a 5.4 MB MP3 file using FFmpeg. This reduced the file size so that I could submit the audio for transcription. I then used OpenAI's Whisper API (`whisper-1`) to convert the podcast into text. The resulting `transcript.txt` contained 16,412 characters.

For the PDF, I used `pypdf` to extract text from the 41-page *Ethics Guidelines for Trustworthy AI*. The extracted text contained 156,411 characters.

After preparing both documents, I applied three chunking strategies:

1. Fixed-size character chunking
2. Recursive character chunking
3. Token-based chunking

I compared chunk counts, average chunk lengths, boundary behavior, and the effects of overlap. I also visualized chunk-size distributions and performed a basic retrieval experiment using the PDF.

The main purpose was to understand the trade-offs between predictable chunk sizes, preservation of textual boundaries, and practical retrieval performance.

## 2. Chunking Experiments

### 2.1 Fixed-size chunking

I used `CharacterTextSplitter` with an empty separator to divide both documents into fixed-length character chunks.

I tested chunk sizes of 500, 1,000, and 2,000 characters with overlaps of 0, 50, and 100 characters.

| Document | Chunk size | Overlap | Number of chunks |
|---|---:|---:|---:|
| Podcast | 500 | 0 | 33 |
| Podcast | 500 | 100 | 41 |
| Podcast | 1,000 | 100 | 19 |
| Podcast | 2,000 | 100 | 9 |
| PDF | 500 | 0 | 313 |
| PDF | 500 | 100 | 391 |
| PDF | 1,000 | 100 | 174 |
| PDF | 2,000 | 100 | 83 |

As expected, smaller chunk sizes produced more chunks. Increasing overlap also increased the number of chunks because some text was repeated between consecutive segments.

The main advantage of this method was its predictable chunk length. However, it did not respect word or sentence boundaries.

For example, the first 500-character podcast chunk ended with:

> Think about the alg

This demonstrated that fixed-size splitting could cut a word in the middle, potentially making the resulting chunk less readable and reducing the context available for retrieval.

Fixed-size chunking is straightforward and useful when consistent lengths are the priority, but it is less suitable when preserving the structure of natural language is important.

### 2.2 Recursive character chunking

Next, I used `RecursiveCharacterTextSplitter`.

Unlike the fixed-size approach, recursive splitting attempts to divide text using a hierarchy of separators, such as paragraphs, line breaks, sentences, and spaces, before falling back to individual characters.

I tested chunk sizes of 500, 1,000, and 2,000 characters with an overlap of 100 characters.

| Document | Chunk size | Overlap | Number of chunks | Average characters |
|---|---:|---:|---:|---:|
| Podcast | 500 | 100 | 42 | 453.0 |
| Podcast | 1,000 | 100 | 19 | 920.1 |
| Podcast | 2,000 | 100 | 9 | 1,859.0 |
| PDF | 500 | 100 | 364 | 443.8 |
| PDF | 1,000 | 100 | 170 | 933.0 |
| PDF | 2,000 | 100 | 83 | 1,895.6 |

The results showed that recursive chunks were generally shorter than their maximum allowed sizes because the splitter attempted to preserve natural text boundaries.

For example, when using 500-character chunks, the podcast's average recursive chunk length was approximately 453 characters, while the PDF's average was approximately 444 characters.

Compared with fixed-size chunking, the recursive approach generally preserved word boundaries better in the chunks I inspected.

This was especially useful for the podcast transcript, where the text consisted primarily of continuous conversational language.

For the PDF, recursive chunking also provided more natural divisions, although the extracted document contained line breaks and formatting artifacts that influenced the splitting process.

Overall, recursive splitting provided a useful balance between controlling chunk length and preserving readability.

### 2.3 Token-based chunking

For the third experiment, I used `TokenTextSplitter` with the `cl100k_base` tokenizer.

I tested chunk sizes of 300, 500, and 1,000 tokens, using a 50-token overlap.

| Document | Chunk size (tokens) | Number of chunks | Average characters |
|---|---:|---:|---:|
| Podcast | 300 | 14 | 1,392.3 |
| Podcast | 500 | 8 | 2,260.9 |
| Podcast | 1,000 | 4 | 4,297.5 |
| PDF | 300 | 127 | 1,477.9 |
| PDF | 500 | 71 | 2,443.3 |
| PDF | 1,000 | 34 | 4,836.4 |

Token-based chunking differs from character-based chunking because tokens are the units used by language models to process text.

A token does not necessarily correspond to one character or one complete word.

This makes token-based chunking particularly useful when preparing text for an LLM with a limited context window.

However, token-based splitting does not automatically preserve complete sentences.

When I inspected the first five podcast chunks produced with a 300-token limit, all five ended in the middle of a sentence.

An important consideration is that the numerical settings are not directly comparable across the strategies. For example, a 500-token chunk generally contains substantially more characters than a 500-character chunk.

Therefore, the differences in chunk counts between token-based and character-based methods should not be interpreted as a direct measure of quality.

### 2.4 Chunk-size distributions

I created histograms comparing fixed-size and recursive chunking for both documents using a 500-character chunk size and 100-character overlap.

The fixed-size approach produced a highly concentrated distribution because most chunks were close to the specified maximum length.

Recursive chunking produced a wider distribution of chunk lengths because it prioritized separator boundaries instead of always filling the maximum character allowance.

The histograms helped illustrate an important trade-off:

- Fixed-size chunking provides more consistent chunk lengths.
- Recursive chunking allows greater variation in order to preserve more natural boundaries.

The visualizations are included in `chunking_strategies.ipynb`.

### 2.5 Boundary-quality experiment

To investigate chunk boundaries, I calculated the percentage of chunks ending with a period, exclamation mark, or question mark.

I used this as a simple proxy for whether chunks ended at sentence boundaries.

The initial results were:

| Document | Fixed-size | Recursive (initial) | Token-based |
|---|---:|---:|---:|
| Podcast | 7.3% | 2.4% | 9.1% |
| PDF | 3.3% | 24.7% | 4.3% |

Initially, the recursive splitter had a surprisingly low punctuation-ending rate for the podcast.

I investigated the separator behavior and tested `keep_separator="end"`.

After this change, the recursive podcast chunks achieved a punctuation-ending rate of 100%, while the PDF remained at 24.7%.

This demonstrated that separator handling can strongly influence the measured result.

However, the punctuation-ending rate has an important limitation: a chunk ending with a period is not necessarily semantically complete, and a chunk without terminal punctuation may still contain a complete idea.

The difference between the podcast and PDF also showed how document formatting affects chunking behavior. The PDF contained many extracted line breaks and formatting irregularities that were not present in the same way in the podcast transcript.

Therefore, punctuation-ending rate should be treated as a basic diagnostic rather than a definitive measurement of semantic quality.

## 3. Retrieval Query

To test how chunking could support information retrieval, I performed a simple keyword-based search over the recursively split PDF.

The query was:

**What are the seven requirements for trustworthy AI?**

I used the following keywords:

- `seven requirements`
- `human agency`
- `robustness`

Each PDF chunk received a score based on how many of these keywords appeared in its text.

This was a simple keyword retrieval experiment rather than a semantic vector search.

The search identified **17 matching chunks**.

The top displayed results were:

| PDF chunk | Keyword score | Relevance |
|---|---:|---|
| 219 | 2 | Contains the complete list of seven requirements |
| 128 | 2 | Contains the first three requirements and explanations |
| 14 | 2 | Contains the complete list of seven requirements |

These results demonstrated that the chunking process preserved passages that could be retrieved to answer a specific question.

However, because the scoring system relied on exact keywords, it did not measure the full semantic relevance of each passage.

## 4. Retrieved Evidence

The most useful retrieved passage was PDF chunk 219, which contained the complete list of seven requirements.

**Retrieved source: PDF Chunk 219 — Score: 2**

**PDF page: 24**

> Key guidance derived from Chapter II:
>
> Ensure that the AI system’s entire life cycle meets the seven key requirements for Trustworthy AI: (1) human agency and oversight, (2) technical robustness and safety, (3) privacy and data g overnance, (4) transparency, (5) diversity, non-discrimination and fairness, (6) environmental and s ocietal well-being and (7) accountability.
>
> Consider technical and non-technical methods to ensure the implementation of those requirements.

The original PDF extraction contained spacing artifacts such as `g overnance` and `s ocietal`. These artifacts were retained in the retrieved text and illustrate a limitation of the document extraction process.

This passage directly supports the query because it explicitly identifies the seven requirements and presents them as an enumerated list.

The retrieval score alone would not be sufficient to establish that the chunk answers the question. The actual content of the retrieved passage provides the necessary evidence.

Another retrieved passage, chunk 14, also contained the complete list, demonstrating that relevant information appeared in more than one part of the document.

The retrieval output was saved in `retrieval_evidence.txt` so the result can be inspected independently.

## 5. Final Answer

**Question: What are the seven requirements for trustworthy AI?**

According to the *Ethics Guidelines for Trustworthy AI*, the seven requirements are:

1. **Human agency and oversight:** AI systems should support human autonomy and remain subject to appropriate human oversight.
2. **Technical robustness and safety:** AI systems should operate reliably and safely, including when facing errors or unexpected situations.
3. **Privacy and data governance:** AI systems should respect privacy and ensure appropriate management of data.
4. **Transparency:** The operation and use of AI systems should be sufficiently transparent and explainable.
5. **Diversity, non-discrimination and fairness:** AI systems should avoid unfair discrimination and support accessibility and fair treatment.
6. **Environmental and societal well-being:** AI systems should take their effects on society and the environment into account.
7. **Accountability:** Appropriate mechanisms should exist to establish responsibility and enable oversight of AI systems.

**Source:** *Ethics Guidelines for Trustworthy AI*, PDF page 24, retrieved chunk 219.

The retrieved passage provides direct evidence for all seven requirements. The short explanations above summarize their general meaning; the retrieved passage itself establishes the seven names.

## 6. Limitations and Lessons Learned

### 6.1 Limitations of fixed-size chunking

Fixed-size chunking can split text in the middle of words or sentences.

The podcast example ending with `"Think about the alg"` demonstrated this limitation directly.

Such boundaries can reduce readability and potentially separate information that should remain together.

### 6.2 Limitations of keyword retrieval

The retrieval experiment relied on three manually selected keywords.

Although it successfully identified relevant passages, this method could miss chunks that expressed the same information using different wording.

For example, a passage discussing human control over AI might be relevant to the query without containing the exact phrase `human agency`.

A semantic embedding-based retrieval system could be explored in future work to address this limitation.

### 6.3 Limitations of retrieval ranking

The scoring method counted keyword matches but did not evaluate the overall quality of the answer contained in each chunk.

Chunks with the same score were ordered by chunk index rather than semantic relevance.

Therefore, the top-ranked result was not necessarily the best result according to a meaningful relevance metric.

In this experiment, chunk 219 happened to contain a complete and directly relevant answer, but this should not be interpreted as proof that the ranking method would consistently produce the best result.

### 6.4 PDF extraction artifacts

The PDF extraction introduced irregular spacing, including `"g overnance"` and `"s ocietal"`.

These artifacts can affect both chunk boundaries and exact keyword matching.

This showed that document preprocessing is important before applying chunking or retrieval methods.

### 6.5 Overlap and efficiency

Increasing overlap helped preserve some context across chunk boundaries, but it also increased the number of chunks.

For example, with fixed-size PDF chunking at 500 characters, increasing overlap from 0 to 100 characters increased the number of chunks from 313 to 391.

More chunks can increase storage requirements and retrieval-processing costs.

Therefore, overlap should be chosen carefully rather than maximized automatically.

### 6.6 Limitations of the evaluation method

The boundary-quality experiment measured whether chunks ended with punctuation.

However, this is only an approximate indicator of text quality.

The 100% punctuation-ending rate observed for the recursive podcast configuration was partly influenced by separator handling. It did not prove that every chunk represented a complete semantic unit.

A more comprehensive evaluation would examine whether important facts, explanations, and relationships remain understandable within each chunk.

## 7. Final Recommendation

Based on the experiments, I would generally choose **recursive character chunking** as the starting strategy for both the podcast transcript and the PDF.

For the podcast, recursive splitting offered a better balance between preserving natural language boundaries and keeping chunks within a defined character limit. This is useful for conversational material, where a fixed character boundary can interrupt a sentence or word.

For the PDF, recursive splitting also provided a practical starting point, although I would first improve text preprocessing to address extraction artifacts and irregular formatting.

If the primary requirement were to stay within an LLM's token budget, I would consider token-based chunking instead. Token-based limits are particularly relevant when managing context windows and model input costs.

Fixed-size character chunking remains useful as a simple baseline, but the observed word-splitting behavior makes it less attractive when readability and context preservation matter.

The retrieval experiment demonstrated that chunking can support a traceable question-answering workflow. The system successfully identified a PDF passage containing all seven requirements for trustworthy AI.

However, this single successful query does not establish that one chunking strategy is universally superior or that the retrieval system performs reliably across different questions.

**My main conclusion is that chunking quality depends on the source material, the intended use, and the evaluation criteria.** Choosing a strategy requires balancing chunk length, context preservation, token constraints, overlap, and retrieval effectiveness.

## 8. Supporting Files

The following files document the implementation and results:

| File | Purpose |
|---|---|
| `chunking_strategies.ipynb` | Python implementation, experiments, statistics, and visualizations |
| `transcript.txt` | Podcast transcription generated using Whisper |
| `retrieval_evidence.txt` | Saved keyword retrieval results and source passages |
| `lab_proof.md` | Summary of the methodology, evidence, findings, and limitations |
| Original PDF | Source document used for extraction, chunking, and retrieval |

Together, these files provide a reproducible record of the workflow and the evidence supporting the conclusions.
