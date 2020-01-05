#for visualization of Entity detection importing displacy from spacy:
import spacy
from spacy import displacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English


# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

nytimes= nlp(u"""New York City on Tuesday declared a public health emergency and ordered mandatory measles vaccinations amid an outbreak, becoming the latest national flash point over refusals to inoculate against dangerous diseases.

At least 285 people have contracted measles in the city since September, mostly in Brooklyn’s Williamsburg neighborhood. The order covers four Zip codes there, Mayor Bill de Blasio (D) said Tuesday.

The mandate orders all unvaccinated people in the area, including a concentration of Orthodox Jews, to receive inoculations, including for children as young as 6 months old. Anyone who resists could be fined up to $1,000.""")

entities=[(i, i.label_, i.label) for i in nytimes.ents]
print(entities)

# displacy.render(nytimes, style = "ent",jupyter = False)