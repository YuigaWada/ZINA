You are given an image, the same reference caption, and an “Original” caption
in which candidate hallucination words have been wrapped with XML tags:

  <object> … </object>
  <spatial_relation> … </spatial_relation>
  <attribute> … </attribute>
  <number> … </number>
  <text> … </text>
  <named_entities_fact> … </named_entities_fact>

For EACH tagged word decide:
  – If it must be corrected, return the corrected word.
  – If it is already correct, return the original word unchanged.

Return ONLY a result “tagged_segment: replacement”, e.g.
  <object>chair</object>: book

────────────────  FEW‑SHOT EXAMPLES  ────────────────

# 1. number: wrong or missing object
Original : There are <number>three</number> cats.  
Reference: Two cats are on the sofa.  
Output   : <number>three</number>: two

# 2. spatial_relation: wrong spatial or positional term
Original : An apple <spatial_relation>on</spatial_relation> the table.  
Reference: An apple is under the table.  
Output   : <spatial_relation>on</spatial_relation>: under

# 3. attribute: wrong adjective or adverb
Original : <attribute>Red</attribute> sky over the mountains.  
Reference: Blue sky over the mountains.  
Output   : <attribute>Red</attribute>: Blue

# 4. object: wrong quantity
Original : There is a <object>chair</object> on the table.  
Reference: There is a book on the table.  
Output   : <object>chair</object>: book

# 5. named_entities_fact: incorrect named entity 
Original : The image shows the <named_entities_fact>John F. Kennedy Center</named_entities_fact>.  
Reference: The image shows the White House.  
Output   : <named_entities_fact>John F. Kennedy Center</named_entities_fact>: White House

# 6. text: incorrect visible text
Original : A sign says <text>'Restaurant'</text>.  
Reference: A sign says 'Hotel'.  
Output   : <text>'Restaurant'</text>: 'Hotel'

────────────  NOW PROCESS THIS SAMPLE  ────────────

Original:  
[Original]

Reference:  
[Reference]

Instructions:  
- Only look at segments already wrapped in XML tags in the Original (`<object>…</object>`, `<spatial_relation>…</spatial_relation>`, etc.).  
- Do **not** add any new tags.  
- Decide for **each** existing tag whether it needs correction, but then choose **only one** tagged segment to report (the first or most obvious error).  
- Output **exactly one** line in the form:  
  `<tag>word</tag>: corrected_word`  
—and nothing else.
