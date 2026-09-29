# Beleggingsdashboard

Lokaal Streamlit-dashboard dat exports van meerdere brokers combineert en je posities en rendement berekent.

## Starten
```
pip install -r requirements.txt
streamlit run app.py      (of dubbelklik start.bat)
```

## Ondersteunde exports
| Broker | Export |
|---|---|
| Trade Republic | Transactie-export (CSV) |
| DEGIRO | Inbox → Rekeningoverzicht → *Account.csv* (niet Transactions.csv) |
| Trading 212 | History → Export CSV |

Upload via de zijbalk en klik **Opslaan**. Bestanden worden in `data/` bewaard; een nieuwe upload onder dezelfde accountnaam vervangt de oude (upload dus altijd de volledige historie).

## Berekeningen
- **Posities**: gemiddelde-kostprijsmethode, transactiekosten zitten in de kostprijs. Alles in EUR (DEGIRO: werkelijke FX-conversie per order).
- **Beleggingsresultaat**: ongerealiseerd + gerealiseerd + netto dividend.
- **TWR**: time-weighted return van de beleggingen (dagelijks, onafhankelijk van stortingen).
- **IRR**: money-weighted (XIRR) over aan-/verkopen, dividenden en huidige waarde.
- Koersen via Yahoo Finance (ISIN → ticker, automatisch). Aan te passen in tab **Tickers** (`tickers.json`, `manual_prices.json`).
- Trade Republic kaartbetalingen tellen als opname.
