# Modeling Second Language Acquisition

### Groups
* 彭賢訓，107753030
* 陳研佑，107753021

### Goal
When learning a second lenguage, it is helpful knowing where a student is likely to make a mistake. We aim to predict the mistakes from translating practices, from the sentences to translate with student and practice metadata.

### demo 
Run the model, if `pred` is not given then will be `test` with suffix `.pred`:
```R
Rscript code/model.R --train data/fr_en.slam.20171218.train --test data/fr_en.slam.20171218.test --pred results/pred
```

Simple evaluation with Python, code from Duolingo:
```py
python3 code/eval.py --pred results/pred  --key data/fr_en.slam.20171218.test.key
```

### data

The data comes from [Duolingo 2018 SLAM Shared task](http://sharedtask.duolingo.com), we use the `fr_en` part, which is from by students who already knows English learning French.

The data looks like this:

```
# user:YjS/mQOx  countries:CA  days:20.133  client:web  session:lesson  format:reverse_translate  time:9
/A/dQZMu0101  Je            PRON    Number=Sing|Person=1|PronType=Prs|fPOS=PRON++                           nsubj        3
/A/dQZMu0102  suis          VERB    Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin|fPOS=VERB++       cop          3
/A/dQZMu0103  sûr           ADJ     Gender=Masc|Number=Sing|fPOS=ADJ++                                      ROOT         0

# user:YjS/mQOx  countries:CA  days:20.136  client:web  session:lesson  format:reverse_translate  time:17
QtTuzqbZ0101  Ce            DET     Gender=Masc|Number=Sing|fPOS=DET++                                      det          2
QtTuzqbZ0102  cheval        NOUN    Gender=Masc|Number=Sing|fPOS=NOUN++                                     nsubj        4
QtTuzqbZ0103  est           VERB    Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|fPOS=VERB++       cop          4
QtTuzqbZ0104  nul           ADJ     Gender=Masc|Number=Sing|fPOS=ADJ++                                      ROOT         0
```

The line starting with `#` indicates an exercise, with session metadata, then follows each word, along with linguistic metadata (generated
 from Google Syntexnet)

We primarily preprocess the dataset in the following ways:
* filter out less than 1 day of usage (user needs to familiarize the app)
* Extract word length and Morphological complexity
* Lowercase tokens, and then factorize them
* Split exercise and token indices
* Replace negative and `NA` time with the median (not the best way)

 
### code

We use logistic regression on the following features:
```
word length (tokenLen)
morphlogical complexity (morphComplex)
part of speech (pos)
dependency label (depLabel)
user
session
format
days
time
token index
exercise index
```
Some of which (e.g. token length, exercise index) is derived from the dataset. The model is simply built by the `glm` call.

### results

The shared task organizers provide a script for evaluating the predictive performance with AUC and F1 scores, we additionally make the ROC curve and double density plot.

Our model's score:
```
Metrics:	accuracy=0.822	avglogloss=0.436	auroc=0.684	F1=0.028
```
Unfortunately, the metrics is worse than the baseline model provided by the shared task organizers, which reports AUC of 0.771.
