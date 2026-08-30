# Clinical Data Model

Versione: 1.0

## Obiettivo

Il Clinical Data Model definisce la struttura standard utilizzata da HLEO
per rappresentare qualsiasi informazione clinica relativa alla tricologia.

Ogni dato estratto deve rispettare questo modello.

---

# Entità principali

## Patient

- Age
- Sex
- Ethnicity (se disponibile)

## Diagnosis

- Disease
- Hair loss type
- Severity
- Stage
- Date of diagnosis

## Symptoms

- Hair shedding
- Hair thinning
- Scalp symptoms
- Duration

## Treatments

- Drug
- Dosage
- Route
- Frequency
- Duration

## Outcomes

- Improvement
- Stable
- Worsening
- Time to response

## Adverse Effects

- Description
- Severity
- Outcome

## Laboratory Findings

- Test
- Value
- Unit
- Reference range

## Evidence

- Source
- Study type
- Evidence level
- DOI / PMID
- URL