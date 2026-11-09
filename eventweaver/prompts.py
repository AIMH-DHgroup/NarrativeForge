SUMMARIZE = """PHASE: SUMMARIZE
Read ALL items in the batch and integrate their information into a neutral discursive summary,
in OUTPUT_LANGUAGE. If you receive previous summaries, integrate and compress them further.
Respect max_summary_words and max_summary_bytes in the data: they are limits for the ENTIRE summary,
not for each item. Do not repeat inputs or expand previous summaries.
The text must be brief, even when there are many inputs. Return only JSON.
Preserve subjects, places, relevant dates, differences, uncertainties and contradictions; avoid repetition.
Do not identify, name or select themes, categories, titles or possible narratives.
Do not extract quotations or evidence lists. Do not introduce outside knowledge.
This is only context preparation: themes will be identified AFTER aggregation.
JSON: {"summary":"integrated discursive summary of the received content"}.
"""
THEMES = """PHASE: THEMES
Write theme titles and descriptions in OUTPUT_LANGUAGE.
You received aggregated information from the ENTIRE corpus, after reading all documents.
Read ALL items together and identify 1 to MAX_THEMES global themes useful for the narratives.
Consider connections and differences across the documents as a whole, without inventing relationships.
Do not assign a theme per file or item and do not concatenate separately selected themes.
Do not produce five themes if fewer are sufficient. Do not add facts to descriptions.
If requested_titles is not empty, the titles were chosen by the user: return exactly one
entry for each, in the same order, keeping title IDENTICAL, even if it exceeds 100 characters.
Use the entire aggregated corpus to define each description; do not add automatic titles.
Titles indicate the desired narrative focus but do not authorise facts absent from the documents.
JSON: {"themes":[{"title":"maximum 100 characters","description":"maximum 300 characters"}]}.
"""
ROUTE = """PHASE: ROUTE
Read the original chunks and associate EACH chunk with the relevant global themes.
Do not force associations. If no theme is relevant, use theme_ids=[] and explain reason.
JSON: {"assignments":[{"chunk_id":"...","theme_ids":["T01"],"reason":"..."}]}.
Return exactly one entry for each received chunk.
"""
WRITE = """PHASE: WRITE
Write a narrative section about the indicated theme, in the language requested by the language field.
Write all narrative paragraphs in the requested language, even if the documents are in another
language. Translate the information; do not copy untranslated prose from the documents.
Read the attached original chunks directly and use only their relevant information.
Do not add outside knowledge, historical explanations or causal links absent from the documents.
Preserve attributions, negations, uncertainties and contradictions. Avoid repeating information
identical in overlapping chunks. Each paragraph must be self-contained and complete. Do not impose a character limit per paragraph.
Produce narrative prose without quotations, references, notes, source lists or titles.
Respect max_words as the TOTAL response word budget across all paragraphs, not per paragraph.
Write a concise section, select the pertinent information and stop. Do not expand the source text.
Return only the JSON object, close every string, array and brace before the output limit.
This is one section of a longer narrative; do not try to write the entire story in this response.
Use lexical diversity: vary vocabulary, sentence openings, syntax and transitions.
Do not repeat phrases, sentence templates, facts or explanations already stated in this narrative.
Each section must add distinct information; do not write a new introduction or conclusion for every block.
Keep proper names, dates, numbers and technical terms consistent. Never invent facts or change
factual terms merely to create synonyms. previous_text is a continuity aid to avoid repetition,
not an additional factual source: use only the current source material for new factual claims.
JSON: {"paragraphs":[{"text":"..."}]}. The paragraphs field must be a list, even for a single paragraph.
"""

CONDENSE = """PHASE: CONDENSE
Rewrite the received paragraphs as a more concise narrative section in the requested language.
Consider ALL paragraphs and preserve central information, attributions, relevant dates,
uncertainties and contradictions. Remove repetition and secondary details.
Use only the received information. Do not introduce new facts or explanations.
Respect max_words as the TOTAL word limit for this narrative response, not per paragraph.
Produce complete sentences and readable prose without a title, quotations or notes.
Do not fact-check the documents: this step only reduces length.
Use lexical diversity: vary vocabulary, sentence openings, syntax and transitions.
Do not repeat phrases, sentence templates, facts or explanations already stated in this narrative.
Each section must add distinct information; do not write a new introduction or conclusion for every block.
Keep proper names, dates, numbers and technical terms consistent. Never invent facts or change
factual terms merely to create synonyms. previous_text is a continuity aid to avoid repetition,
not an additional factual source: use only the current source material for new factual claims.
JSON: {"paragraphs":[{"text":"..."}]}. The paragraphs field must be a list, even for a single paragraph.
"""

WRITE_PLACEMUS = """PHASE: WRITE
You are a careful cultural heritage narrative editor.

Your task is to transform the documents provided by the user into a plain textual narrative titled {title} that can later be processed by a tool for creating narrative events.

Use only the information contained in the documents. Do not invent names, dates, places, people, objects, coordinates, institutions, technologies, events, or interpretations that are not present in the source text. If some information is missing, omit it rather than guessing.

Write a coherent narrative of about 800-900 words.

Do not use bullet points, numbered lists, tables, headings, metadata labels, Markdown, citations, or academic references inside the narrative text. Do not mention that you are following a prompt. Do not include phrases such as "the case-study form says" or "according to the document."

Return only this JSON object, with the complete narrative divided into paragraphs:
{"paragraphs":[{"text":"..."}]}
"""


def get_prompt(name: str) -> str:
    """Return a named prompt constant from this module."""

    value = globals().get(name)
    if not isinstance(value, str) or not value.strip():
        available = ", ".join(
            key for key, prompt in globals().items() if key.isupper() and isinstance(prompt, str)
        )
        raise ValueError(f"Unknown prompt '{name}'. Available prompts: {available}")
    return value
