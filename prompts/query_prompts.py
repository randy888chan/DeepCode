RESOURCE_IDENTIFICATION_PROMPT = """You are an expert in identifying resources included in a given text, following the guidelines below:
<GUIDELINES>
[1] The input will be a text that may contain various resources.
[2] A resource can be either a URL, or a file path.
[3] A URL starts with "http://" or "https://", while a file path typically starts with a drive letter (e.g., "C:\") or a Unix-style path (e.g., "/home/user/").
[4] **ONLY return a valid JSON dict[str, List[str]]** with the same structure as the example below, where "resources" is a list containing the identified resources **in the order they appear in the text**.
[5] If no resources are found, return an empty list [].
</GUIDELINES>

<EXAMPLE>
"Check out this link: https://example.com and compare with C:\\documents\\file.txt."

```json
{
    "resources": [
        "https://example.com",
        "C:\\documents\\file.txt"
        ]
}
```
</EXAMPLE>
"""

RESOURCE_TYPE_JUDGMENT_PROMPT = ""
QUERY_TYPE_IDENTIFICATION_PROMPT = ""
QUERY_AUGMENTATION_PROMPT = ""
LOOP_QUERY_AUGMENTATION_PROMPT = ""