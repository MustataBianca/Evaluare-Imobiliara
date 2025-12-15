Acest proiect implementează un model de Regresie Ridge (Metoda Celor Mai Mici Pătrate Stabilizată - CMMP Stabilizat) pentru a estima prețul apartamentelor. Proiectul se 
bazează pe un set restrâns de date și folosește tehnici de pre-procesare și regularizare pentru a obține predicții stabile, care evită prețurile negative.



Arhitectura Proiectului

  Fișierul implementare.py definește setul de date de antrenare și setul de date de test
  Fișierul main.py încarcă aceste date, antrenează modelul și evaluează rezultatele pe baza datelor de test, afișând și o comparație a erorilor
  Fișierul interfata.py creează o aplicație interactivă pentru utilizator

Metodologia de Modelare

Pentru modelarea datelor programul nostru folosește următoarele librării standard: numpy, pandas și sklearn
Setul de date de antrenare folosește atributele a 50 de anunțuri reale, acestea fiind: suprafața utilă în metrii pătrați, numărul de camere, distanța față de centru, anul construirii, prezența mobilierului, etajul, numărul de băi și starea generală

Pre-procesarea Datelor

  Scalare:
    Datele sunt standardizate (scalate) pentru a normaliza distribuția caracteristicilor, prevenind astfel ca variabilele cu valori mari (ex: Suprafața) să domine modelul.
    
  Adăugarea Interceptului:
    O coloană de 1 este adăugată ca primă coloană la matricea scalată (A_final), permițând modelului să învețe un termen liber (intercept).

Modelul de Regresie

Este folosită regresia Ridge. Modelul este antrenat cu un parametru de regularizare $\alpha = 100.0$. Această valoare mare introduce o penalizare semnificativă asupra magnitudinii coeficienților, stabilizând soluția și prevenind supra-adaptarea pe setul mic de date.

Evaluarea din main

Fișierul main.py calculează și afișează o comparație detaliată a predicțiilor modelului pe cele 4 proprietăți din setul de test.
Metrici de performanță utilizate:
  Eroarea Absolută (EURO): Eroare = |Preț estimat - Preț real|
  Eroarea Procentuală (%): Eroare = (Eroare absoluta/Preț real)*100

Interfața Utilizator

Aplicația Streamlit permite utilizatorului să interacționeze cu modelul antrenat.
  Utilizatorul introduce valorile pentru cele 8 caracteristici.
  Funcția predict_price preia datele de intrare, aplică transformările de pre-procesare (calculul vârstei, scalarea), efectuează predicția și afișează prețul estimat, rotunjit la cel mai apropiat EURO
  Funcția predict_price preia datele de intrare, aplică transformările de pre-procesare (calculul vârstei, scalarea), efectuează predicția și afișează prețul estimat, rotunjit la cel mai apropiat EURO
