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


CULTURAL_HERITAGE_SHORT_PROMPT_TEMPLATE = """Transform the case-study form into a concise plain-text cultural-heritage narrative.

Use only the source information. Do not invent or infer missing facts.

Write 250-350 words in 3-5 short, complete paragraphs. Each paragraph should express one clear narrative event or idea.

Include only the essential context, places, music or heritage elements, and digital or public-experience details when present.

Do not use headings, bullet points, lists, tables, Markdown, citations, metadata labels, or explanations.

Output only the final narrative.

CASE-STUDY FORM:
{case_study_text}
"""


CULTURAL_HERITAGE_DETAILED_PROMPT_TEMPLATE = """You are a careful cultural heritage narrative editor.

Transform the case-study form into a detailed plain-text narrative that can later be processed into narrative events.

Use only information contained in the source form. Do not invent names, dates, places, people, objects, coordinates, institutions, technologies, musical practices, events, causal explanations, relationships, meanings, or interpretations that are not present in the source. If information is missing, uncertain, unclear, or only implied, omit it rather than guessing.

Write a coherent narrative of about 650-800 words. Divide the narrative into 7-10 short paragraphs. Treat each paragraph as one narrative event or one coherent narrative unit. Each paragraph must be syntactically complete, contain complete sentences, and end with proper punctuation. Do not split sentences between paragraphs.

Preserve the factual meaning of the source as closely as possible. Rephrase only when needed to create a readable narrative. Keep important names, places, dates, institutions, project names, musical genres, instruments, repertoires, technologies, heritage objects, accessibility information, and public objectives when they are available.

The narrative should cover, when present:
the title or core concept of the itinerary;
the geographical, territorial, or landscape context;
the historical chronology or period;
the main places, sites, buildings, monuments, landscapes, and routes;
the movement, sequence, or relationship between places;
the musical genres, musical practices, instruments, performers, repertoire, soundscapes, or acoustic elements;
the cultural, historical, social, political, or religious significance of the music and places;
the tangible and intangible heritage objects involved;
the preservation, transformation, destruction, reuse, accessibility, or current condition of places;
the digital, immersive, acoustic, XR, VR, AR, auralization, spatial-audio, reconstruction, app-based, or visitor-interface layer;
the intended educational, promotional, public-engagement, inclusivity, accessibility, or social objectives;
the collaborations, institutions, or project framework when relevant.

If the source describes multiple places, organize the narrative as a detailed journey through those places. If the source describes one main place, organize the narrative as a detailed exploration moving from context to heritage, music, experience, technology, and public meaning.

Do not use headings, bullet points, numbered lists, tables, metadata labels, JSON, Markdown, citations, academic references, or explanatory notes. Do not mention that you are following a prompt. Do not include phrases such as "the case-study form says" or "according to the document."

Write in a neutral, informative cultural-heritage style. Avoid promotional language, emotional exaggeration, fictional storytelling, and unsupported interpretation.

Output only the final narrative.

CASE-STUDY FORM:
{case_study_text}
"""


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


CSV_SHORT_PROMPT_TEMPLATE = """Transform the CSV value-chain record into a concise plain-text narrative.

Use only the source information. Do not invent or infer missing facts. Preserve exact numeric values and units when used.

Write 200-300 words in 3-4 short, complete paragraphs.

Focus only on the essential territory, value chain, assets, challenges, innovation, relevance, and key indicators.

Do not use headings, bullet points, lists, tables, Markdown, citations, metadata labels, or explanations.

Output only the final narrative.

CSV VALUE-CHAIN RECORD:
{source_text}
"""


CSV_DETAILED_PROMPT_TEMPLATE = """You are a careful narrative editor for mountain value-chain case studies.

Transform the CSV value-chain record into a detailed plain-text narrative.

Use only the information contained in the CSV record. Do not invent names, dates, places, institutions, products, actors, challenges, innovations, landscape features, protected areas, numeric values, percentages, statistics, causal explanations, comparisons, trends, or interpretations that are not present in the source record. If information is missing, uncertain, unclear, or only implied, omit it rather than guessing.

Write a coherent narrative of about 500-700 words. Divide the narrative into 6-9 short paragraphs. Each paragraph must express one clear narrative unit, such as territorial context, socioeconomic context, environmental context, land-use system, value-chain description, local assets, challenges, innovation, project relevance, or mountain-landscape significance. Each paragraph must contain complete sentences and end with proper punctuation.

Preserve the factual meaning of the source record as closely as possible. Keep important names, territories, administrative units, products, services, institutions, value-chain descriptors, innovation types, land-use systems, protected areas, indicators, numeric values, percentages, and units exactly as written when possible.

The narrative should cover, when present:
the value-chain descriptor;
the member state, region, district, sub-region, NUTS level, LAU, or mountain reference landscape;
the product, service, activity, or value chain described;
the land-use system;
the local assets;
the key challenges;
the type of value chain;
the type of innovation;
the brief description of the innovation;
what the innovation is linked to;
the reasons why the value chain is relevant;
the synthetic description of the value chain;
the protected areas, landscape features, or environmental context;
the socioeconomic, territorial, demographic, economic, agricultural, environmental, or land-use indicators;
any numeric values or percentages from selected numeric columns that are relevant to the narrative.

When using quantitative information:
preserve exact numeric values and percentages as written in the source record;
preserve the original units when they are provided;
do not invent units;
do not round, approximate, normalize, convert, or recalculate values;
do not calculate new totals, averages, rates, percentages, or rankings unless they are already provided in the source record;
do not infer trends unless the source record explicitly supports them;
do not compare values unless the comparison is directly supported by the record;
combine related indicators into readable contextual sentences rather than listing them mechanically;
do not omit numeric values that are important for understanding the selected indicators.

Organize the narrative so that the reader first understands the territory and value chain, then the assets and challenges, then the innovation and relevance. If the record contains strong quantitative context, integrate it naturally into the relevant paragraphs.

Do not use headings, bullet points, numbered lists, tables, metadata labels, JSON, Markdown, citations, academic references, or explanatory notes. Do not mention that you are following a prompt. Do not include phrases such as "the CSV record says" or "according to the dataset."

Write in a neutral, informative, research-oriented style. Avoid promotional language, emotional exaggeration, fictional storytelling, and unsupported interpretation.

Output only the final narrative.

CSV VALUE-CHAIN RECORD:
{source_text}
"""


DOCX_PROMPT_TEMPLATES = {
    "standard": GENERAL_PROMPT_TEMPLATE,
    "short": CULTURAL_HERITAGE_SHORT_PROMPT_TEMPLATE,
    "detailed": CULTURAL_HERITAGE_DETAILED_PROMPT_TEMPLATE,
}

CSV_PROMPT_TEMPLATES = {
    "standard": CSV_VALUE_CHAIN_PROMPT_TEMPLATE,
    "short": CSV_SHORT_PROMPT_TEMPLATE,
    "detailed": CSV_DETAILED_PROMPT_TEMPLATE,
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
            raise ValueError(
                f"Unknown prompt strategy: {prompt_strategy}. "
                f"Available strategies for cultural-heritage: {', '.join(DOCX_PROMPT_STRATEGIES)}."
            )
        return

    if kind == "value-chain":
        if strategy not in CSV_PROMPT_TEMPLATES:
            raise ValueError(
                f"Unknown prompt strategy: {prompt_strategy}. "
                f"Available strategies for value-chain: {', '.join(CSV_PROMPT_STRATEGIES)}."
            )
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
