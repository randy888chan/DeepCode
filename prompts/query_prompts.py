RESOURCE_IDENTIFICATION_PROMPT = """You are an expert in identifying resources included in a given text, following the guidelines below:
<GUIDELINES>
[1] The <INPUT> will be a text that may contain various resources.
[2] A resource can be either a URL, or a file path.
[3] A URL starts with "http://" or "https://", while a file path typically starts with a drive letter (e.g., "C:\") or a Unix-style path (e.g., "/home/user/").
[4] **ONLY return a valid JSON dict[str, list[dict[str, str]]]** with the same structure as the example below, where "resources" is a list containing the identified resources **in the order they appear in the text**.
[5] If no resources are found, return an empty list [].
</GUIDELINES>

<EXAMPLE>
<INPUT>: "Check out this link: https://example.com and compare with C:\\documents\\file.txt."

<OUTPUT>: ```json
{{
    "resources": [
        {{
            "path": "https://example.com",
            "type": "url"
        }},
        {{
            "path": "C:\\documents\\file.txt",
            "type": "file"
        }}
    ]
}}
```
</EXAMPLE>

<INPUT>: {user_input}

<OUTPUT>: """

QUERY_TYPE_IDENTIFICATION_PROMPT = """You need to identify the type of query based on the provided text, following these guidelines:
<GUIDELINES>
[1] The <QUERY> will be a string, representing a task.
[2] The <RESOURCES> will be a list[dict[str, str]] containing resources identified from the query.
[3] <HAS RESOURCE> is a boolean indicating whether the query contains any valid resource.
[4] The <QUERY> can be classified into one of the following types: "paper_reproduction", "chat_based_coding".
   - "paper_reproduction" means the query is related to reproducing a research paper, and there is supposed to be at least one resource in the <RESOURCES>.
   - "chat_based_coding" means the query is related to coding tasks, and there may or may not be resources in the <RESOURCES>.
[5] **ONLY return a valid JSON str, either "paper_reproduction" or "chat_based_coding"**.

<EXAMPLE>
<QUERY>: "I want to reproduce the paper here: https://arxiv.org/abs/2401.00001.
<RESOURCES>: [
    {{
        "path": "https://arxiv.org/abs/2401.00001",
        "type": "url"
    }}
]
<HAS RESOURCE>: true

<OUTPUT>: ```json
"paper_reproduction"
```
</EXAMPLE>

<QUERY>: {user_input}
<RESOURCES>: {resources}
<HAS RESOURCE>: {has_resource}
<OUTPUT>: """


QUERY_AUGMENTATION_PROMPT = """You are an expert in augmenting queries based on the provided text, following these guidelines:
[1] The <USER INPUT> will be a string, probabily representing a brief task.
[2] Firstl, you need to identify the ultimate <GOAL> of this query, which is supposed to be a brief str. This could be a coding project (e.g. "Implement a web scraper in Python"), a technical programming task (e.g. "Implement an MCP server which enables duckduckgo search"), ...
[3] Next, based on the <USER INPUT> and <GOAL>, you need to list several <ITEMS> that are necessary to further identify the requirements, functionalities, and constraints of the <GOAL>. Each item should be a **proposition** that can be answered with a "yes" or "no". For example, if the <GOAL> is "Implement a web scraper in Python", the items could be:
- "The web scraper is aimed at scraping movie reviews from reddit."
- "The web scraper should be able to handle pagination."
- "The web scraper will not need to handle multi-modality data."
[4] **ONLY return your a valid JSON dict**.

<EXAMPLE>
<INPUT>: "I want to implement a web scraper in Python."
<OUTPUT>: ```json
{{
    "query": "I want to implement a web scraper in Python.",
    "goal": "Implement a web scraper in Python",
    "items": [
        "The web scraper is aimed at scraping movie reviews from reddit.",
        "The web scraper should be able to handle pagination.",
        "The web scraper will not need to handle multi-modality data.",
        "The web scraper has the following interfaces for adjusting the scraping parameters: \n1. `max_pages`: The maximum number of pages to scrape.\n2. `max_reviews_per_page`: The maximum number of reviews to scrape per page.\n3. `output_format`: The format of the output file (e.g., JSON, CSV).\n4. `scrape_interval`: The interval between scraping requests to avoid being blocked by the website.\n5. `user_agent`: The user agent string to use for the requests."
    ]
}}
```
</EXAMPLE>

<INPUT>: {user_input}
<OUTPUT>: """


LOOP_QUERY_AUGMENTATION_PROMPT = """You are an expert in augmenting queries based on the provided context (including <USER INPUT>, <GOAL> and <DETAILS>), following these guidelines:
[1] The <user_input> is the original query, which is a string representing a task.
[2] The <GOAL> is a brief string representing the ultimate goal of the query.
[3] <DETAILS> contain the items that are necessary to further identify the requirements, functionalities, and constraints of the <GOAL>. And each item has been confrimed by the user.
[4] Here, your task is to adjust the <GOAL> and <DETAILS> based on the user's response, and arrange them into <OUTPUT> following the example below.
[5] **ONLY return a valid JSON dict**.

<EXAMPLE>
<USER INPUT>: "I want to implement a web scraper in Python to get movie reviews."
<GOAL>: "Implement a web scraper in Python to get movie reviews"
<DETAILS>: [
    "The web scraper is aimed at scraping movie reviews from reddit.: yes",
    "The web scraper should be able to handle pagination.: yes",
    "The web scraper will not need to handle multi-modality data.: no, the user says that the web scraper should also handle images and videos.",
    "The web scraper has the following interfaces for adjusting the scraping parameters: \n1. `max_pages`: The maximum number of pages to scrape.\n2. `max_reviews_per_page`: The maximum number of reviews to scrape per page.\n3. `output_format`: The format of the output file (e.g., JSON, CSV).: yes"
    ]

<OUTPUT>: ```json
{{
    "query": "I want to implement a web scraper in Python to get movie reviews.",
    "goal": "Implement a web scraper in Python to get multi-modality movie reviews",
    "items": [
        "The web scraper will should also handle images and videos.",
        "The higest resolution of the images and videos should be 1080p.",
        ...
        ],
    "confirmed_items": [
        "The web scraper is aimed at scraping movie reviews from reddit.",
        "The web scraper should be able to handle pagination.",
        "The web scraper has the following interfaces for adjusting the scraping parameters: \n1. `max_pages`: The maximum number of pages to scrape.\n2. `max_reviews_per_page`: The maximum number of reviews to scrape per page.\n3. `output_format`: The format of the output file (e.g., JSON, CSV)."
    ]
}}
```
</EXAMPLE>

<USER INPUT>: {user_input}
<GOAL>: {goal}
<DETAILS>: {details}
<OUTPUT>: """