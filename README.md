# Évaluation de la Toxicité & Conformité AI Act  
Analyse du dataset RealToxicityPrompts + Évaluation du modèle Llama 3.1 8B

---

## Objectif du projet

Ce projet vise à :

1. **Analyser la toxicité des prompts** du dataset RealToxicityPrompts (AllenAI).  
2. **Évaluer le comportement du modèle Llama 3.1 8B** face à des demandes potentiellement dangereuses.  
3. **Mesurer la conformité du modèle** aux exigences du AI Act européen.

Projet réalisé par une équipe de **4 étudiants** de la Licence SID — Université de Toulouse.

---

## Méthodologie

### 1) Analyse du dataset
- Nettoyage et normalisation des données  
- Étude de la toxicité des prompts et réponses  
- Analyse des prompts *challenging*  
- Tests statistiques : corrélation, Student, proportions  

### 2) Évaluation du modèle LLM (Llama 3.1 8B)
Pour chaque prompt :
- génération d’une réponse  
- calcul d’un score de toxicité  
- détection d’un refus approprié  
- analyse de la sécurité et des alternatives proposées  
- comparaison toxicité prompt → réponse  

### 3) Critères évalués
- **Toxicité faible**  
- **Refus approprié** des demandes illicites  
- **Non‑amplification** de la toxicité  
- **Alternatives sûres**  
- **Transparence** du modèle  

---

## Articles du AI Act concernés

- **Article 5** — Pratiques interdites  
- **Article 10** — Gestion des données et évaluation des risques  
- **Article 52** — Obligations des GPAI  
- **Article 53** — GPAI à risque systémique  

---

## Structure du projet
```
ai-act-evaluation-rtp/
│
├── data/
│   ├── raw/
│   │   └── prompts.jsonl
│   └── processed/
│       └── rtp_propre.csv
│
├── src/
│   ├── pretraitement/
│   │   └── nettoyage_donnees.py
│   ├── analyse/
│   │   ├── graphiques.py
│   │   └── tests_statistiques.py
│   ├── evaluation/
│   │   └── evaluation_llm.py
│   └── pipeline.py
│
├── requirements.txt
└── README.md
```
