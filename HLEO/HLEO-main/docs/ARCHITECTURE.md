# HLEO Architecture

Versione: 1.0

## Data Flow

User
    │
    ▼
FastAPI API
    │
    ▼
Search Manager
    │
    ▼
Search Providers
    │
    ▼
Crawler
    │
    ▼
Parser
    │
    ▼
Clinical Extractor
    │
    ▼
Clinical Validator
    │
    ▼
Knowledge Base
    │
    ▼
Clinical Reasoner
    │
    ▼
API Response

---

## Modules

### API
Gestisce tutte le richieste REST.

### Search
Ricerca le fonti.

### Crawler
Scarica il contenuto delle pagine.

### Parser
Pulisce e normalizza il testo.

### Clinical Extractor
Estrae dati clinici strutturati.

### Clinical Validator
Verifica che i dati siano coerenti con il Clinical Schema.

### Knowledge Base
Memorizza i dati strutturati.

### Clinical Reasoner
Collega le informazioni provenienti da fonti diverse.

---

## Design Principles

- Modularità
- Tracciabilità
- Evidenze verificabili
- Nessuna informazione inventata
- Separazione tra estrazione e ragionamento