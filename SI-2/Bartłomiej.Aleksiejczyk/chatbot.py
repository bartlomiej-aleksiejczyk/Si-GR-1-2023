
import pandas as pd
import gradio as gr
from os.path import exists



with gr.Blocks() as chatbot:
    def zapytanie_chatbot(zapytanie):
        tresc_zapytania = (zapytanie[-1][0]).split(";")
        komenda = tresc_zapytania[0]
        if komenda == "wersy":
            if len(tresc_zapytania) >= 3:
                plik = tresc_zapytania[1]
            else:
                zapytanie[-1][1] = "Błąd: za mało podanych atrybutów komendy"
                return zapytanie
            if (tresc_zapytania[2]).isdigit():
                wartosc = int(tresc_zapytania[2])
                print("numer")
            else:
                zapytanie[-1][1] = "Błąd: wartość liczbowa nie jest numeryczna"
                print(tresc_zapytania[2])
                return zapytanie
            if exists(plik):
                with open(plik) as tabela:
                    tabela = tabela.readlines()
                    if wartosc > len(tabela):
                        zapytanie[-1][1] = "Błąd: wartość jest większa od całkowitej liczby wierszy ({})".format(
                            len(tabela))
                        return zapytanie
                    else:
                        atrybuty = len(tabela[0].split(","))
                        wersy = "Informacje ogólne: ilosc obiektow: {}, ilosc atrybutow {} \n".format(len(tabela),
                                                                                                      atrybuty)
                        tabela = tabela[:wartosc]
                        for wers in tabela:
                            wersy += wers
                        odpowiedz = wersy
            else:
                odpowiedz = "Błąd: plik o podanej nazwie nie istnieje"
        elif komenda == "Decyzyjne":
            if len(tresc_zapytania) >= 2:
                plik = tresc_zapytania[1]
                if exists(plik):
                    csv = pd.read_csv(plik, sep=" ", header=None)
                    odpowiedz = f"Ilość klas decyzyjnych: {len(csv.iloc[0]) - 1}"
                else:
                    odpowiedz = "Błąd: plik o podanej nazwie nie istnieje"
            else:
                zapytanie[-1][1] = "Błąd: za mało podanych atrybutów komendy"
                return zapytanie
        elif komenda == "Decyzyjne_wielkosc":
            if len(tresc_zapytania) >= 2:
                plik = tresc_zapytania[1]
                if exists(plik):
                    wersy = "Wielkość poszczególnych klas decyzyjnych: \n"
                    csv = pd.read_csv(plik, sep=" ", header=None)
                    ilosc_klas = len(csv.iloc[0]) - 1
                    for klasa in range(0, ilosc_klas):
                        wersy += str(csv[klasa].value_counts())
                        wersy += "\n"
                    odpowiedz = wersy
                else:
                    odpowiedz = "Błąd: plik o podanej nazwie nie istnieje "
            else:
                zapytanie[-1][1] = "Błąd: za mało podanych atrybutów komendy"
                return zapytanie
        elif tresc_zapytania == "przyklady":
            pass
        else:
            odpowiedz = "Witaj, ten czatbot sluży do pozyskania podstawowych danych z tabeli bez nagłówków w formacie txt. \n " \
                        "Aby wyświelić n pierwszych wersów pliku, oraz informację o pliku, wpisz do czatu komendę (bez cudzysłowów): \"wersy;<ścieżka do pliku, bez nawiasów>;<ilość wersów, bez nawiasów>\" \n " \
                        "Aby wyświelić ilość atrybutów decyzyjnych: \"Decyzyjne;<ścieżka do pliku, bez nawiasów>\" \n "\
                        "Aby wyświelić wielkość poszczególnych klas decyzyjnych: \"Decyzyjne_wielkosc;<ścieżka do pliku, bez nawiasów>\" \n "
        zapytanie[-1][1] = odpowiedz
        return zapytanie
    def user(wiadomosc, history):
        return "", history + [[wiadomosc, None]]
    czatbot = gr.Chatbot()
    wiadomosc = gr.Textbox()
    clear = gr.Button("Clear")

    wiadomosc.submit(user, [wiadomosc, czatbot], [wiadomosc, czatbot], queue=False).then(
        zapytanie_chatbot, czatbot, czatbot,
    )
    clear.click(lambda: None, None, czatbot, queue=False)
chatbot.launch()
