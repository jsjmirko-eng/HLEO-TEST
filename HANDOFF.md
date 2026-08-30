# HANDOFF — HLEO project checkpoint

Questo file è il checkpoint ufficiale per riprendere il lavoro nella nuova conversazione.

---

1) STATO GIT

- Branch corrente: `main`
- Commit HEAD locale: `6e6bd0229d86a9897e20fe107d861ed89df871fb` (`checkpoint: final verified state`)
- Commit del checkpoint precedente presente sul remoto: `c73868e02505a015a9e720fee06ddcbbffb234f8`
  - Verifica eseguita con `git ls-remote` sul remote `origin`.
- Il lavoro corrente include ancora un insieme di artefatti/test untracked, non funzionali al checkpoint.

File untracked presenti nell'albero di lavoro — sono artefatti di test e NON fanno parte del checkpoint funzionale:

- `integration_test_compare.py`
- `node_modules/`
- `package-lock.json`
- `package.json`
- `playwright_smoke_test.py`
- `playwright_tests/`
- `real_assistant_compare_test.py`
- `server_real_test.log`
- `server_real_test2.log`
- `tokenize.js`

---

2) OBIETTIVO DEL PROGETTO

HLEO deve mantenere separati i contesti Scientific e RWE. L'Assistant e la funzione Compare devono usare correttamente i profili provenienti dalle ricerche appropriate senza mescolare i contesti.

---

3) MODIFICHE FUNZIONALI GIÀ VERIFICATE

- Modifica frontend in `templates/index.html` nella funzione `runCompare()`.
- Il confronto ora costruisce il contesto dell'Assistant/Compare usando:
  - il contesto attivo;
  - la snapshot Scientific;
  - la snapshot RWE;
  - `clinical_profile_episode_ids`;
  - `rwe_profile_episode_ids`.
- La logica unisce i contesti e unifica gli ID senza duplicati.
- Questo mantiene disponibili all'Assistant i profili delle due ricerche correnti senza trasformarli in persistenza permanente.

---

4) TEST GIÀ SUPERATI (sintesi)

- Scientific Assistant: PASS
  - query: `finasteride alopecia`
  - `articles_count = 43`
  - risultati scientifici presenti
  - PMID `42396141`
  - PMID `42466626`
  - Assistant con contesto Scientific corretto
  - nessun contesto RWE usato

- RWE Assistant: PASS
  - query: `finasteride hair shedding`
  - `rwe_evidence_count = 25`
  - esempi di episode_id:
    - `140084`
    - `1271403`
    - `12538`
    - `139223`
  - Assistant con contesto RWE corretto
  - nessun contesto Scientific usato

- Mixing context: PASS
  - Scientific → RWE → Scientific
  - RWE → Scientific
  - il contesto corretto viene recuperato dalle snapshot per-mode
  - nessun mixing rilevato

- Experiences: PASS
  - la funzione/API risponde correttamente
  - il precedente `0 records` era dovuto al database di verifica vuoto, non a un errore della funzione

---

5) DATABASE — RISULTATO DELL'ULTIMA INVESTIGAZIONE

Database attualmente usato dal server/browser:
- file: `/workspace/project/HLEO-official/test_hleo_browser_verify.db`
- DATABASE_URL: `sqlite:///./test_hleo_browser_verify.db`

Questo database è vuoto per:
- ClinicalProfile: `0`
- RWEProfile: `0`
- PatientExperience: `0`

Database con dati di test:
- file: `/workspace/project/HLEO-official/test_hleo_real.db`
- ClinicalProfile: `9`
- RWEProfile: `25`
- PatientExperience: `0`

Altri DB test/snapshot rimasti:
- `test_hleo_browser_verify.db` — DB corrente del server/browser, vuoto per le tabelle di interesse
- i vecchi snapshot `test_hleo_real.db`, `test_hleo_browser.db`, `test_hleo_real2.db` sono stati rimossi come artefatti non necessari alla sessione corrente

Chiarimento: il DB del server/browser e il DB con dati di test erano DB diversi.

---

6) PERSISTENZA

Verificato che:
- `GET /search` non salva automaticamente i risultati nel DB;
- `POST /pipeline/run` estrae i profili e li mantiene nel contesto/risultato, senza persistenza automatica;
- `POST /rwe/extract` e `POST /rwe/extract-batch` non persistono automaticamente i risultati della ricerca;
- `POST /experiences/ingest` non deve essere interpretato come persistenza automatica della ricerca;
- le scritture DB esistenti relative a session/chat, admin/partner o operazioni esplicite sono separate dalla persistenza automatica dei risultati di ricerca.

Questa architettura è corretta.

---

7) INTERPRETAZIONE

- Non è stato dimostrato che i dati siano stati persi.
- Il browser test sta interrogando un DB vuoto.
- I risultati Scientific/RWE verificati precedentemente erano disponibili nel contesto della ricerca e non implicano automaticamente che siano stati persistiti nel DB.
- La persistenza automatica dei risultati di ricerca NON deve essere introdotta senza una decisione esplicita.

---

8) DECISIONI / VINCOLI

- Non introdurre automaticamente persistence dei risultati Scientific/RWE.
- Non modificare `DB/models`.
- Non modificare il backend solo per far passare un test.
- Non cambiare il comportamento già verificato.
- Prima di qualsiasi modifica futura, verificare se il comportamento richiesto è davvero persistenza oppure contesto temporaneo.

---

9) PROSSIMI PASSI

Il prossimo agente deve:
1. leggere `HANDOFF.md`;
2. verificare `git status` e `HEAD`;
3. non ricostruire il lavoro precedente da zero;
4. non modificare nulla immediatamente;
5. riprendere dall'analisi della persistenza/DB e decidere con l'utente se:
   - A) il browser test deve essere eseguito sul DB corretto/popolato, oppure
   - B) la persistenza dei profili deve essere implementata come requisito applicativo.

Non assumere A o B senza verifica.

---

10) REGOLA FONDAMENTALE

Questo HANDOFF.md è il punto di ripartenza. Il prossimo agente deve usarlo per continuare il lavoro, non per ricostruirlo da zero.

(END)
