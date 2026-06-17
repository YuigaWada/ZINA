You are given an image, a generated caption (Original), and a human reference caption (Reference).
Your task is to detect any single words in the Original that are NOT supported by the image
(hallucinations) and label each of them with exactly one tag from:

  • object              – wrong or missing object
  • spatial_relation    – wrong spatial or positional term
  • attribute           – wrong adjective or adverb
  • number              – wrong quantity
  • text                – incorrect visible text
  • named_entities_fact – incorrect named entity

Return ONLY a comma‑and‑slash separated list of “word, tag” pairs, e.g.  
  three, number / apples, object  
If no hallucinations exist, return exactly:  
  none

────────────────  FEW‑SHOT EXAMPLES  ────────────────

# 1. number
Original : "There are three cats."
Reference: "Two cats are on the sofa."
Output   : three, number

# 2. spatial_relation
Original : "An apple on the table."
Reference: "An apple is under the table."
Output   : on, spatial_relation

# 3. attribute
Original : "Red sky over the mountains."
Reference: "Blue sky over the mountains."
Output   : Red, attribute

# 4. object
Original : "There is a chair on the table."
Reference: "There is a book on the table."
Output   : chair, object

# 5. named_entities_fact
Original : "The image shows the John F. Kennedy Center."
Reference: "The image shows the White House."
Output   : John F. Kennedy Center, named_entities_fact

# 6. text
Original : "A sign says 'Restaurant'."
Reference: "A sign says 'Hotel'."
Output   : Restaurant, text

────────────  NOW PROCESS THIS SAMPLE  ────────────

Original:
[Original]

Reference:
[Reference]

Output:
