#!/bin/bash

# --- CONFIGURAZIONE ---
# Il branch di default che stai utilizzando (hai risolto con 'main')
BRANCH_NAME="main"

# --- FUNZIONI ---

# Controlla che la directory corrente sia un repository Git
check_git_repo() {
    if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        echo "ERRORE: Non sei all'interno di un repository Git."
        exit 1
    fi
}

# Esegue l'azione di push
perform_push() {
    echo "--- Esecuzione del PUSH verso origin/$BRANCH_NAME ---"
    
    # Esegue il push. L'opzione -u non è necessaria qui perché l'hai già impostata.
    git push origin "$BRANCH_NAME"
    
    if [ $? -eq 0 ]; then
        echo "✅ AGGIORNAMENTO COMPLETATO: Commit caricati su GitHub."
    else
        echo "❌ ERRORE: Fallimento durante il push. Controlla la tua connessione o le credenziali SSH/PAT."
    fi
}

# --- LOGICA PRINCIPALE ---

check_git_repo

# 1. Controlla lo stato dei file
git status
echo "-----------------------------------------------"

# Chiede all'utente il messaggio di commit
read -r -p "Inserisci il messaggio per il commit (es. 'Aggiunto il modulo X'): " COMMIT_MESSAGE

if [ -z "$COMMIT_MESSAGE" ]; then
    echo "❌ Operazione annullata: il messaggio di commit è obbligatorio."
    exit 1
fi

echo "--- Aggiunta di tutti i file modificati/nuovi (git add .) ---"
git add .

echo "--- Creazione del commit con il messaggio: \"$COMMIT_MESSAGE\" ---"
# -m: specifica il messaggio
# --quiet: esegue il commit senza mostrare tutti i dettagli dei file
git commit -m "$COMMIT_MESSAGE" --quiet

# Controlla se il commit è stato effettivamente creato (non ci sono state modifiche)
if [ $? -ne 0 ]; then
    # Git commit -m non restituisce 0 se non ci sono modifiche da committare
    echo "⚠️ NESSUNA MODIFICA: Non ci sono file nuovi o modificati da caricare. Uscita."
else
    # 3. Esegue il Push
    perform_push
fi

echo "-----------------------------------------------"