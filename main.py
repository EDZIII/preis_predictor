import streamlit as st
import pickle
import numpy as np
import pandas as pd


def formatTreibstoff(t):
    if t == 'Benzin':
        return '0'
    elif t == 'Diesel':
        return '1'
    else:
        return '2'


def formatAufbau(aufbau):
    if aufbau == 'Kombi':
        return '0'
    elif aufbau == 'Limousine':
        return '1'
    elif aufbau == 'Van / Kleinbus':
        return '2'
    elif aufbau == 'SUV':
        return '3'
    elif aufbau == 'Cabriolet / Sport':
        return '4'
    elif aufbau == 'Coupé':
        return '5'
    else:
        return '0'


def formatGetriebe(getriebe):
    if getriebe == 'Automat':
        return '0'
    else:
        return '1'


dataDir = r'data/df.csv'
# load the model and dataframe
df = pd.read_csv(dataDir )
pipe = pickle.load(open(r'data/pipe.pkl', "rb"))
st.title("Preisvorhersage für Gebrauchtwagen in der Schweiz")

# marke
marken = df['MARKE'].unique()
marke_auswahl = st.selectbox('Marke', marken)

# modell
modellen = df['MODELL'].loc[df['MARKE'] == marke_auswahl]
modell_auswahl = st.selectbox('Modell', modellen.unique())

# plz
plz = st.selectbox('Plz', df['PLZ'].unique())

# regdate
#regdate=df['REGDATE'].unique().astype(int)
regdate_auswahl = st.slider('Erstzulassung', min_value=1960,max_value=2022)

# km stand
km_stand = ["5'000 - 9'999",
            "35'000 - 39'999",
            "50'000 - 54'999",
            "70'000 - 74'999",
            "75'000 - 79'999",
            "90'000 - 94'999",
            "150'000 - 159'999",
            "400'000 - 449'999"
            "450'000 - 499'999",
            "500'000+",
            ]
km_stand_auswahl = st.selectbox('KM Stand', km_stand)

# leistung
leistung_ps = st.number_input('Leistung PS', value=100)

# farbe
farbe = st.selectbox('Farbe', df['FARBE'].unique())

# treibstoff
treibstoff = st.selectbox('treibstoff', ['Benzin', 'Diesel', 'Hybrid'])

# aufbau
aufbau = st.selectbox('Aufbau', ['Kombi', 'Limousine', 'Van / Kleinbus', 'SUV', 'Cabriolet / Sport', 'Coupé'])

# getriebe
getriebe = st.selectbox('Getriebe Typ', ['Automat', 'Schaltgetriebe'])

# türen
tueren = st.selectbox('Türen', df['TUEREN'].unique())

# Prediction
if st.button('Preis Vorhersage'):

    query = np.array(
        [int(plz), int(tueren), marke_auswahl, modell_auswahl, int(regdate_auswahl), km_stand_auswahl, int(leistung_ps), farbe,
         int(formatTreibstoff(treibstoff)),
         int(formatAufbau(aufbau)), int(formatGetriebe(getriebe))])
    query = query.reshape(1, 11)
    prediction = str(int(np.exp(pipe.predict(query)[0])))

    st.title("Die Preisvorhersage für diese Konfiguration ist " + prediction)
