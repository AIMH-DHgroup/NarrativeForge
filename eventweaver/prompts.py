GENERAL_PROMPT_TEMPLATE = """You are a careful cultural heritage narrative editor.

Your task is to transform the case-study form provided by the user into a plain textual narrative that can later be processed by a tool for creating narrative events.

Use only the information contained in the case-study form. Do not invent names, dates, places, people, objects, coordinates, institutions, technologies, events, or interpretations that are not present in the source text. If some information is missing, omit it rather than guessing.

Write a coherent narrative of about 450-550 words. The narrative must be divided into short paragraphs. Each paragraph must express one clear narrative event, place, transition, historical moment, musical experience, technological experience, social context, or heritage-related idea.

For this task, treat a paragraph as equivalent to a narrative event. Each paragraph must be syntactically complete and must contain complete sentences. Do not split sentences between paragraphs. Do not end a paragraph with an incomplete sentence.

Preserve the factual meaning of the source text as closely as possible. Rephrase only when needed to make the text readable as a narrative. Do not add explanatory comments, external knowledge, or general background information.

The narrative should normally include, when available in the source form:
the title or main concept of the itinerary;
the geographical or territorial context;
the historical period or chronology;
the main places, sites, buildings, landscapes, or routes;
the musical genre, musical practices, instruments, performers, repertoire, or sound-related elements;
the cultural, historical, social, or religious significance of the music and places;
the movement or relationship between the places in the itinerary;
the tangible and intangible heritage objects involved;
the accessibility, preservation, or transformation of the places;
the planned digital, immersive, acoustic, or technological experiences;
the intended public, educational, promotional, or social objectives.

If the source form describes multiple places, organize the narrative as a journey through those places. If the source form describes one main place, organize the narrative as a progressive exploration of that place, from context to music, heritage, experience, and technology.

Do not use bullet points, numbered lists, tables, headings, metadata labels, JSON, Markdown, citations, or academic references. Do not mention that you are following a prompt. Do not include phrases such as "the case-study form says" or "according to the document."

Write in an informative, neutral, cultural-heritage style. Avoid promotional language, emotional exaggeration, and fictional storytelling.

Output only the final narrative.

CASE-STUDY FORM:
{case_study_text}
"""


CULTURAL_HERITAGE_SHORT_PROMPT_TEMPLATE = """You are a careful cultural heritage narrative editor.

Transform the case-study form into a concise plain-text narrative that can later be processed as narrative events.

Use only the information contained in the source form. Do not invent names, dates, places, people, objects, institutions, technologies, musical practices, events, or interpretations. If information is missing, omit it.

Write about 300-400 words in 4-6 short paragraphs. Treat each paragraph as one narrative event. Each paragraph must contain complete sentences and must not end abruptly.

Prioritize only the most important available information:
the itinerary or main concept;
the geographical or territorial context;
the main places, buildings, routes, or landscapes;
the musical, cultural, historical, or social significance;
the digital, immersive, acoustic, accessibility, educational, or public-engagement layer, if present.

If the source describes multiple places, organize the narrative as a compact journey. If it describes one main place, organize it as a compact exploration from context to heritage meaning and visitor experience.

Do not use headings, bullet points, numbered lists, tables, metadata labels, JSON, Markdown, citations, or academic references. Do not mention the source form or the prompt.

Write in a neutral, informative cultural-heritage style. Output only the final narrative.

CASE-STUDY FORM:
{case_study_text}
"""


CULTURAL_HERITAGE_DETAILED_PROMPT_TEMPLATE = """You are a careful cultural heritage narrative editor.

Transform the case-study form into a detailed plain-text narrative that can later be processed into narrative events.

Use only information contained in the source form. Do not invent names, dates, places, people, objects, coordinates, institutions, technologies, musical practices, events, causal explanations, or interpretations that are not present in the source. If information is missing, omit it rather than guessing.

Write about 600-750 words in 7-10 short paragraphs. Treat each paragraph as one narrative event or one coherent narrative unit. Each paragraph must be syntactically complete and must contain complete sentences.

Preserve the factual meaning of the source as closely as possible. Rephrase only to create a readable narrative. Keep important names, places, dates, institutions, musical genres, instruments, technologies, objects, accessibility information, and project objectives when they are available.

The narrative should cover, when present:
the title or core concept of the itinerary;
the geographical and territorial context;
the historical chronology;
the main places, sites, buildings, landscapes, and routes;
the movement or relation between places;
the musical genres, musical practices, instruments, performers, repertoire, or sound-related elements;
the cultural, historical, social, political, or religious significance;
the tangible and intangible heritage objects involved;
the state of preservation, transformation, destruction, or accessibility of places;
the digital, immersive, acoustic, XR, VR, AR, auralization, spatial audio, or technological layer;
the educational, promotional, public-engagement, inclusivity, or social objectives;
the collaborations, institutions, or project framework if they are relevant.

If the source describes multiple places, organize the narrative as a detailed journey. If it describes one main place, organize it as a detailed exploration from context to heritage, music, experience, technology, and public meaning.

Do not use headings, bullet points, numbered lists, tables, metadata labels, JSON, Markdown, citations, or academic references. Do not mention the source form or the prompt.

Write in a neutral, informative cultural-heritage style. Avoid promotional language, emotional exaggeration, and fictional storytelling.

Output only the final narrative.

CASE-STUDY FORM:
{case_study_text}
"""


CULTURAL_HERITAGE_STRICT_PROMPT_TEMPLATE = """You are a careful cultural heritage narrative editor.

Transform the case-study form into a plain-text narrative suitable for later narrative-event processing.

Use only source information. Do not invent or infer names, dates, places, people, institutions, coordinates, objects, technologies, musical practices, events, purposes, or interpretations. If a detail is not explicitly present in the source, do not include it.

Write 450-600 words in 5-8 paragraphs. Do not exceed 600 words. Each paragraph must represent one clear narrative event or narrative unit. Each paragraph must be syntactically complete and end with proper punctuation.

The narrative must include the most important available information:
the itinerary or main concept;
the geographical context;
the historical context;
the main places, routes, sites, buildings, or landscapes;
the musical, sonic, cultural, historical, social, religious, or political dimension;
the digital, immersive, acoustic, accessibility, educational, or public-engagement layer if present.

Preserve important names, places, dates, institutions, musical genres, instruments, technologies, and project names exactly as written when possible.

Do not use headings. Do not use bullet points. Do not use numbered lists. Do not use tables. Do not use Markdown. Do not use JSON. Do not use metadata labels. Do not use citations. Do not include explanations. Do not say "according to the document" or "the case-study form says."

Output only the final narrative in plain prose.

CASE-STUDY FORM:
{case_study_text}
"""


CULTURAL_HERITAGE_EVENT_FOCUSED_PROMPT_TEMPLATE = """You are a cultural heritage narrative editor preparing text for narrative-event extraction.

Transform the case-study form into a plain-text narrative composed of event-like paragraphs.

Use only the information contained in the source form. Do not add external knowledge, invented facts, unsupported dates, fictional transitions, or interpretations not present in the source.

Write about 450-550 words in 6-9 short paragraphs. Each paragraph must function as one narrative event. A narrative event may describe a place, a transition, a historical moment, a musical practice, a cultural function, a technological experience, an accessibility condition, or a public-engagement objective.

Each paragraph must be complete, self-contained, and syntactically correct. Do not split a sentence across paragraphs. Do not end a paragraph with an incomplete sentence.

Organize the narrative in a clear event sequence. If the source describes an itinerary, follow the spatial or chronological movement between places. If the source describes one place, move from territorial context to historical background, musical or cultural meaning, heritage objects, digital or immersive experience, accessibility, and public relevance.

Preserve source facts, names, places, dates, institutions, objects, technologies, musical genres, and project names when available. Rephrase only when necessary for readability.

Do not use headings, bullet points, numbered lists, tables, labels, JSON, Markdown, citations, or academic references. Do not mention the prompt or the source form.

Write in a neutral cultural-heritage style. Output only the final narrative.

CASE-STUDY FORM:
{case_study_text}
"""


CULTURAL_HERITAGE_FAITHFULNESS_FIRST_PROMPT_TEMPLATE = """You are a careful cultural heritage narrative editor.

Transform the case-study form into a plain-text narrative while preserving the source information as faithfully as possible.

Use only information explicitly present in the case-study form. Do not add background knowledge, explanations, interpretations, assumptions, or invented links between facts. If a fact is uncertain, missing, implicit, or unclear, omit it.

Write about 450-550 words in short paragraphs. Each paragraph should express one coherent narrative unit and must contain complete sentences.

Keep the wording close to the source when possible, especially for:
names of places;
names of people;
institutions and project names;
dates and historical periods;
musical genres, instruments, repertoires, and practices;
technologies, digital assets, XR tools, acoustic methods, or immersive devices;
heritage objects, documents, sites, and monuments;
accessibility, preservation, or public-engagement information.

Do not compress the source into generic statements if specific information is available. Do not replace precise source terms with vague synonyms. Do not introduce cause-effect relations unless they are present in the source.

The narrative should still be readable and coherent. If there are multiple places, organize them as a journey. If there is one place, organize the narrative as a progressive exploration of context, heritage, music, experience, technology, and public relevance.

Do not use headings, bullet points, numbered lists, tables, labels, JSON, Markdown, citations, or academic references. Do not mention the source form or the prompt.

Output only the final narrative.

CASE-STUDY FORM:
{case_study_text}
"""


CULTURAL_HERITAGE_DIGITAL_HERITAGE_FOCUSED_PROMPT_TEMPLATE = """You are a careful cultural heritage narrative editor with expertise in digital heritage.

Transform the case-study form into a plain-text narrative suitable for later narrative-event processing.

Use only the information contained in the source form. Do not invent technologies, digital assets, methods, institutions, places, musical practices, or interpretations.

Write about 450-600 words in 5-8 short paragraphs. Each paragraph must express one clear narrative event or narrative unit and must contain complete sentences.

The narrative should preserve the cultural, musical, historical, and territorial context, but it should pay particular attention to digital heritage information when present, including:
3D models and digital reconstructions;
XR, VR, AR, immersive rooms, games, apps, or web applications;
spatial audio, soundscape recording, acoustic measurement, auralization, impulse responses, or sound diffusion;
digital assets, repositories, metadata, interoperability, or ECCCH-oriented workflows;
touch screens, QR codes, mobile devices, headphones, loudspeakers, or visitor interfaces;
accessibility, inclusivity, public mediation, educational use, and remote access.

If these digital or technological elements are absent from the source, do not invent them. In that case, focus on the available cultural-heritage information.

If the source describes multiple places, organize the narrative as a journey that connects places, heritage meanings, and digital experiences. If it describes one place, organize it from context to heritage significance, musical or cultural experience, digital mediation, accessibility, and public relevance.

Do not use headings, bullet points, numbered lists, tables, metadata labels, JSON, Markdown, citations, or academic references. Do not mention the source form or the prompt.

Write in a neutral, informative cultural-heritage style. Output only the final narrative.

CASE-STUDY FORM:
{case_study_text}
"""


DOCX_PROMPT_TEMPLATES = {
    "standard": GENERAL_PROMPT_TEMPLATE,
    "short": CULTURAL_HERITAGE_SHORT_PROMPT_TEMPLATE,
    "detailed": CULTURAL_HERITAGE_DETAILED_PROMPT_TEMPLATE,
    "strict": CULTURAL_HERITAGE_STRICT_PROMPT_TEMPLATE,
    "event_focused": CULTURAL_HERITAGE_EVENT_FOCUSED_PROMPT_TEMPLATE,
    "faithfulness_first": CULTURAL_HERITAGE_FAITHFULNESS_FIRST_PROMPT_TEMPLATE,
    "digital_heritage_focused": CULTURAL_HERITAGE_DIGITAL_HERITAGE_FOCUSED_PROMPT_TEMPLATE,
    "concise": CULTURAL_HERITAGE_SHORT_PROMPT_TEMPLATE,
}


CSV_VALUE_CHAIN_PROMPT_TEMPLATE = """You are a careful narrative editor for mountain value-chain case studies.

Your task is to transform the CSV value-chain record provided by the user into a short plain-text narrative.

Use only the information contained in the CSV record. Do not invent names, dates, places, institutions, products, actors, challenges, innovations, landscape features, numeric values, percentages, statistics, or interpretations that are not present in the source record. If information is missing, omit it rather than guessing.

The CSV record may contain numeric values, percentages, indicators, counts, areas, population values, economic values, environmental values, or other quantitative information. When numeric data are present in the selected numeric columns, you must preserve the original values and include them in the narrative. Do not round, approximate, normalize, convert, or recalculate numeric values unless an aggregated value is already provided in the CSV record.

If several selected numeric columns describe the same territory, value chain, landscape, socioeconomic context, land-use system, protected area, or environmental condition, aggregate them narratively. This means that you should combine the relevant quantitative indicators into one or more readable sentences that explain the context of the value chain. Preserve every numeric value or percentage that you use in the aggregation.

For example, if the source record contains population, area, elevation, protected-area share, employment, agricultural surface, or other socioeconomic or territorial indicators, describe them together as contextual evidence. Do not list numbers mechanically. Explain how the selected indicators characterize the territory or value chain, while remaining strictly faithful to the record.

Write a coherent narrative of about 300-500 words. The narrative must be divided into short paragraphs. Each paragraph must express one clear narrative unit, such as the territorial context, the value chain, the local assets, the challenges, the innovation, the relevance for the MOVING project, the mountain landscape, or the quantitative socioeconomic and territorial context.

For this task, treat a paragraph as equivalent to a narrative event. Each paragraph must be syntactically complete and must contain complete sentences. Do not split sentences between paragraphs. Do not end a paragraph with an incomplete sentence.

The narrative should normally include, when available in the source record:
the value-chain descriptor;
the member state, region, district, or mountain landscape;
the product, service, activity, or value chain described;
the land-use system;
the local assets;
the key challenges;
the type of value chain;
the type of innovation;
the brief description of the innovation;
the reasons why the value chain is relevant;
the synthetic description of the value chain;
the protected areas or mountain reference landscape, if available;
important socioeconomic, territorial, demographic, economic, agricultural, environmental, or land-use indicators, if available;
any numeric values or percentages from selected numeric columns that are relevant to the narrative.

When using quantitative information:
preserve the exact numeric values and percentages as written in the source record;
keep the original units when they are provided;
do not invent units;
do not infer trends unless the source record explicitly supports them;
do not compare values unless the comparison is directly supported by the record;
do not calculate new totals, averages, rates, percentages, or rankings unless they are already present in the source record;
do not omit numeric values that are important for understanding the selected indicators.

Do not use bullet points, numbered lists, tables, headings, metadata labels, JSON, Markdown, citations, or academic references. Do not mention that you are following a prompt. Do not include phrases such as "the CSV record says" or "according to the dataset."

Write in an informative, neutral, research-oriented style. Avoid promotional language, emotional exaggeration, and fictional storytelling.

Output only the final narrative.

CSV VALUE-CHAIN RECORD:
{source_text}
"""


CSV_NUMERIC_AWARE_PROMPT_TEMPLATE = """You are a careful narrative editor for mountain value-chain case studies.

Your task is to transform the CSV value-chain record provided by the user into a short plain-text narrative.

Use only the information contained in the CSV record. Do not invent names, dates, places, institutions, products, actors, challenges, innovations, landscape features, numeric values, percentages, statistics, or interpretations that are not present in the source record. If information is missing, omit it rather than guessing.

The CSV record may contain numeric values, percentages, indicators, counts, areas, population values, economic values, environmental values, or other quantitative information. Preserve the exact values and keep their units when present. Do not round, approximate, normalize, convert, or recalculate numeric values unless the source record explicitly provides an aggregated value.

If several selected numeric columns describe the same territory, value chain, landscape, socioeconomic context, land-use system, protected area, or environmental condition, aggregate them narratively. Combine the relevant quantitative indicators into readable sentences that explain the context of the value chain while preserving every numeric value or percentage you use.

Write a coherent narrative of about 300-500 words. The narrative must be divided into short paragraphs. Each paragraph must express one clear narrative unit, such as the territorial context, the value chain, the local assets, the challenges, the innovation, the relevance for the MOVING project, the mountain landscape, or the quantitative socioeconomic and territorial context.

Each paragraph must be syntactically complete and must contain complete sentences. Do not split sentences between paragraphs.

Do not use bullet points, numbered lists, tables, headings, metadata labels, JSON, Markdown, citations, or academic references.

Output only the final narrative.

CSV VALUE-CHAIN RECORD:
{source_text}
"""


CSV_FIELD_COVERAGE_PROMPT_TEMPLATE = """You are a careful narrative editor for mountain value-chain case studies.

Your task is to transform the CSV value-chain record provided by the user into a short plain-text narrative.

Use only the information contained in the CSV record. Do not invent names, dates, places, institutions, products, actors, challenges, innovations, landscape features, or interpretations that are not present in the source record. If information is missing, omit it rather than guessing.

Prioritize coverage of the value-chain descriptor, member state, region or mountain landscape, challenges, value-chain type, innovation type, innovation description, land-use systems, local assets, synthetic description, reasons for selection, protected areas, and socioeconomic indicators.

Write a coherent narrative of about 300-500 words. The narrative must be divided into short paragraphs. Each paragraph must express one clear narrative unit.

Each paragraph must be syntactically complete and must contain complete sentences. Do not split sentences between paragraphs.

Do not use bullet points, numbered lists, tables, headings, metadata labels, JSON, Markdown, citations, or academic references.

Output only the final narrative.

CSV VALUE-CHAIN RECORD:
{source_text}
"""


CSV_CONCISE_PROMPT_TEMPLATE = """You are a careful narrative editor for mountain value-chain case studies.

Your task is to transform the CSV value-chain record provided by the user into a concise plain-text narrative.

Use only the information contained in the CSV record. Do not invent names, dates, places, institutions, products, actors, challenges, innovations, landscape features, numeric values, percentages, statistics, or interpretations that are not present in the source record. If information is missing, omit it rather than guessing.

Write a coherent narrative of about 250-350 words. The narrative must be divided into 3-5 short paragraphs.

Preserve only the most important fields and indicators: the value-chain descriptor, the territorial context, the key challenges, the type of value chain, the innovation, the local assets, and the synthetic description.

Do not use bullet points, numbered lists, tables, headings, metadata labels, JSON, Markdown, citations, or academic references.

Output only the final narrative.

CSV VALUE-CHAIN RECORD:
{source_text}
"""


CSV_TERRITORIAL_CONTEXT_PROMPT_TEMPLATE = """You are a careful narrative editor for mountain value-chain case studies.

Your task is to transform the CSV value-chain record provided by the user into a short plain-text narrative.

Use only the information contained in the CSV record. Do not invent names, dates, places, institutions, products, actors, challenges, innovations, landscape features, numeric values, percentages, statistics, or interpretations that are not present in the source record. If information is missing, omit it rather than guessing.

Emphasize the member state, region, district or sub-region, NUTS levels, LAU, mountain landscape, land-use systems, protected areas, and socioeconomic or environmental indicators.

Write a coherent narrative of about 300-500 words. The narrative must be divided into short paragraphs.

Do not use bullet points, numbered lists, tables, headings, metadata labels, JSON, Markdown, citations, or academic references.

Output only the final narrative.

CSV VALUE-CHAIN RECORD:
{source_text}
"""


CSV_INNOVATION_FOCUSED_PROMPT_TEMPLATE = """You are a careful narrative editor for mountain value-chain case studies.

Your task is to transform the CSV value-chain record provided by the user into a short plain-text narrative.

Use only the information contained in the CSV record. Do not invent names, dates, places, institutions, products, actors, challenges, innovations, landscape features, numeric values, percentages, statistics, or interpretations that are not present in the source record. If information is missing, omit it rather than guessing.

Emphasize the type of innovation, the brief description of the innovation, what it is linked to, the challenges addressed, the local assets, and the reasons for selection.

Write a coherent narrative of about 300-500 words. The narrative must be divided into short paragraphs.

Do not use bullet points, numbered lists, tables, headings, metadata labels, JSON, Markdown, citations, or academic references.

Output only the final narrative.

CSV VALUE-CHAIN RECORD:
{source_text}
"""


DOCX_PROMPT_TEMPLATES = {
    "standard": GENERAL_PROMPT_TEMPLATE,
    "short": CULTURAL_HERITAGE_SHORT_PROMPT_TEMPLATE,
    "detailed": CULTURAL_HERITAGE_DETAILED_PROMPT_TEMPLATE,
    "strict": CULTURAL_HERITAGE_STRICT_PROMPT_TEMPLATE,
    "event_focused": CULTURAL_HERITAGE_EVENT_FOCUSED_PROMPT_TEMPLATE,
    "faithfulness_first": CULTURAL_HERITAGE_FAITHFULNESS_FIRST_PROMPT_TEMPLATE,
    "digital_heritage_focused": CULTURAL_HERITAGE_DIGITAL_HERITAGE_FOCUSED_PROMPT_TEMPLATE,
    "concise": CULTURAL_HERITAGE_SHORT_PROMPT_TEMPLATE,
}

CSV_PROMPT_TEMPLATES = {
    "standard": CSV_VALUE_CHAIN_PROMPT_TEMPLATE,
    "numeric_aware": CSV_NUMERIC_AWARE_PROMPT_TEMPLATE,
    "field_coverage": CSV_FIELD_COVERAGE_PROMPT_TEMPLATE,
    "concise": CSV_CONCISE_PROMPT_TEMPLATE,
    "territorial_context": CSV_TERRITORIAL_CONTEXT_PROMPT_TEMPLATE,
    "innovation_focused": CSV_INNOVATION_FOCUSED_PROMPT_TEMPLATE,
}

DOCX_PROMPT_STRATEGIES = list(DOCX_PROMPT_TEMPLATES.keys())
CSV_PROMPT_STRATEGIES = list(CSV_PROMPT_TEMPLATES.keys())


def list_prompt_strategies(prompt_kind: str | None = None) -> list[str]:
    if prompt_kind is None:
        return sorted(set(DOCX_PROMPT_STRATEGIES + CSV_PROMPT_STRATEGIES))
    kind = (prompt_kind or "").strip().lower()
    if kind == "cultural-heritage":
        return list(DOCX_PROMPT_STRATEGIES)
    if kind == "value-chain":
        return list(CSV_PROMPT_STRATEGIES)
    raise ValueError(f"Unknown prompt kind: {prompt_kind}")


def validate_prompt_strategy(prompt_kind: str, prompt_strategy: str) -> None:
    kind = (prompt_kind or "").strip().lower()
    strategy = (prompt_strategy or "standard").strip().lower()
    if kind == "cultural-heritage":
        if strategy not in DOCX_PROMPT_TEMPLATES:
            if strategy in {"numeric_aware", "field_coverage", "territorial_context", "innovation_focused"}:
                raise ValueError(f"{prompt_strategy} is only valid for value-chain CSV prompts.")
            raise ValueError(f"Unknown prompt strategy: {prompt_strategy}. Available strategies for cultural-heritage: {', '.join(DOCX_PROMPT_STRATEGIES)}.")
        return
    if kind == "value-chain":
        if strategy not in CSV_PROMPT_TEMPLATES:
            if strategy in {"short", "detailed", "strict", "event_focused", "faithfulness_first", "digital_heritage_focused"}:
                raise ValueError(f"{prompt_strategy} is only valid for cultural-heritage DOCX prompts.")
            raise ValueError(f"Unknown prompt strategy: {prompt_strategy}. Available strategies for value-chain: {', '.join(CSV_PROMPT_STRATEGIES)}.")
        return
    raise ValueError(f"Unknown prompt kind: {prompt_kind}")


def get_prompt_template(prompt_kind: str, prompt_strategy: str) -> str:
    kind = (prompt_kind or "auto").strip().lower()
    strategy = (prompt_strategy or "standard").strip().lower()
    if kind == "auto":
        raise ValueError("Prompt kind must be resolved before selecting a template.")
    validate_prompt_strategy(kind, strategy)
    if kind == "cultural-heritage":
        return DOCX_PROMPT_TEMPLATES[strategy]
    return CSV_PROMPT_TEMPLATES[strategy]


prompt_template_for = get_prompt_template
