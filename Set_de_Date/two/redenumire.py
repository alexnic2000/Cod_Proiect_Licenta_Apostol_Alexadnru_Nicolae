import os

def redenumește_fișiere_wav(director):
    # Verificăm dacă directorul există
    if not os.path.exists(director):
        print(f"Directorul '{director}' nu există.")
        return
    
    # Listăm fișierele WAV din director
    fișiere_wav = [f for f in os.listdir(director) if f.lower().endswith('.wav')]

    # Redenumește fișierele WAV în stilul dorit
    for i, fișier in enumerate(fișiere_wav):
        nume_nou = f'g5341341_nohash_{i}.wav'
        cale_veche = os.path.join(director, fișier)
        cale_nouă = os.path.join(director, nume_nou)
        os.rename(cale_veche, cale_nouă)
        print(f"Redenumire: {fișier} -> {nume_nou}")

# Specifică calea către directorul cu fișierele WAV
director_lucru = '/Repozitoriulicentaonoff/nientwodataset/two'  # Înlocuiește cu calea directorului tău

# Apelăm funcția pentru a redenumi fișierele
redenumește_fișiere_wav(director_lucru)
