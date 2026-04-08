## General notes
- All indices are derived from **daily data**:
  - TX: daily maximum temperature
  - TN: daily minimum temperature
  - TG/TM: daily mean temperature
  - RR: daily precipitation
- Percentile-based indices use a **reference climatology (typically 1961–1990)** with a 5-day window.
- Outputs are typically **annual or monthly aggregates**.
## resources
* list from https://www.climdex.org/learn/indices/
* some literature:
    * https://journals.ametsoc.org/view/journals/clim/18/11/jcli3366.1.xml
    * https://journals.ametsoc.org/view/journals/clim/26/13/jcli-d-12-00383.1.xml
    * https://pubmed.ncbi.nlm.nih.gov/25546282/
    * https://www.mdpi.com/1660-4601/12/1/227
---

# 1. Temperature Threshold Indices

## FD (Frost Days)
- Core idea: count freezing nights
- Definition: number of days where TN < 0°C
- Data: daily minimum temperature
- Interpretation: cold climate severity
- Resources: Climdex FD

## SU (Summer Days)
- Core idea: count hot days
- Definition: TX > 25°C
- Data: daily max temperature
- Interpretation: heat exposure

## ID (Icing Days)
- Core idea: persistent freezing
- Definition: TX < 0°C
- Interpretation: severe cold daytime conditions

## TR (Tropical Nights)
- Core idea: warm nights
- Definition: TN > 20°C
- Interpretation: heat stress (especially human health)

---

# 2. Growing Season / Degree Days

## GSL (Growing Season Length)
- Core idea: length of biologically active season
- Definition:
  - Start: ≥6 consecutive days with TG > 5°C
  - End: ≥6 consecutive days with TG < 5°C (after mid-year)
- Data: daily mean temperature
- Interpretation: agricultural productivity window

## GDDgrown (Growing Degree Days)
- Core idea: accumulated heat for growth
- Formula: sum(TG - base) for TG > base
- Interpretation: crop development potential

## HDDheatn (Heating Degree Days)
- Core idea: heating demand
- Formula: sum(base − TG) for TG < base
- Interpretation: energy demand (heating)

## CDDcoldn (Cooling Degree Days)
- Core idea: cooling demand
- Formula: sum(TG − base) for TG > base
- Interpretation: energy demand (cooling)

---

# 3. Absolute Temperature Extremes

## TXx (Max Tmax)
- Core idea: hottest day
- Definition: max(TX)
- Interpretation: extreme heat

## TNx (Max Tmin)
- Core idea: warmest night

## TXn (Min Tmax)
- Core idea: coldest day

## TNn (Min Tmin)
- Core idea: coldest night

---

# 4. Percentile-Based Temperature Indices

## TN10p
- Core idea: cold nights frequency
- Definition: % of days TN < 10th percentile
- Interpretation: cold extremes

## TX10p
- Core idea: cool days frequency

## TN90p
- Core idea: warm nights frequency

## TX90p
- Core idea: hot days frequency

- Data: daily temps + climatological percentiles
- Interpretation:
  - ↑ TN90p/TX90p → warming extremes
  - ↑ TN10p/TX10p → cooling extremes

---

# 5. Spell Duration Indices

## WSDI (Warm Spell Duration Index)
- Core idea: prolonged heat
- Definition:
  - ≥6 consecutive days where TX > 90th percentile
- Output: number of days in such spells
- Interpretation: heatwave persistence

## CSDI (Cold Spell Duration Index)
- Core idea: prolonged cold
- Definition:
  - ≥6 consecutive days where TN < 10th percentile
- Interpretation: cold wave persistence

## TXbdTNbd (Cold day/night spells)
- Core idea: compound cold extremes
- Definition:
  - ≥d consecutive days where TX and TN < 5th percentile

---

# 6. Mean and Derived Temperature Metrics

## DTR (Diurnal Temperature Range)
- Core idea: daily variability
- Formula: TX − TN (averaged)
- Interpretation: cloudiness, climate variability

## ETR (Extreme Temperature Range)
- Core idea: intra-period extremes
- Formula: TXx − TNn

## Means
- TMm: mean daily mean temperature
- TXm: mean daily max temperature
- TNm: mean daily min temperature

---

# 7. Temperature Threshold Counts (custom thresholds)

## TMge5 / TMlt5
- Days with TG ≥ / < 5°C

## TMge10 / TMlt10
- Days with TG ≥ / < 10°C

## TXge30 / TXge35
- Hot extremes count

## TNlt2 / TNlt-2 / TNlt-20
- Cold thresholds

## TXgt50p
- % of days above median

---

# 8. Precipitation Intensity & Totals

## PRCPTOT
- Core idea: total rainfall
- Definition: sum(RR ≥ 1 mm)
- Interpretation: total wet-day precipitation

## SDII (Simple Daily Intensity Index)
- Core idea: rainfall intensity
- Formula: total precipitation / number of wet days

---

# 9. Heavy Precipitation Indices

## R10mm / R20mm / Rnnmm
- Core idea: heavy rain frequency
- Definition: count of days RR ≥ threshold

## Rx1day
- Core idea: extreme daily rainfall
- Definition: max(RR)

## Rx5day
- Core idea: multi-day extreme rainfall
- Definition: max 5-day accumulated precipitation

---

# 10. Precipitation Percentile Indices

## R95p
- Core idea: very wet days total
- Definition: sum(RR > 95th percentile)

## R99p
- Core idea: extremely wet days total

## R95pTOT / R99pTOT
- Core idea: contribution to total rainfall
- Formula:
  - 100 × (R95p / PRCPTOT)
  - 100 × (R99p / PRCPTOT)

---

# 11. Wet/Dry Spell Indices

## CDD (Consecutive Dry Days)
- Core idea: drought duration
- Definition: max consecutive RR < 1 mm

## CWD (Consecutive Wet Days)
- Core idea: persistent rainfall
- Definition: max consecutive RR ≥ 1 mm

---

# 12. Drought Indices

## SPI (Standardized Precipitation Index)
- Core idea: precipitation anomaly
- Method:
  - fit distribution to precipitation
  - standardize to normal distribution
- Interpretation:
  - negative → drought
  - positive → wet conditions

## SPEI (Standardized Precipitation Evapotranspiration Index)
- Core idea: drought including evapotranspiration
- Data: precipitation + temperature (PET)
- Interpretation: climate-change-sensitive drought metric

---

# 13. Heatwave Indices (EHF / percentile-based)

## HWN (Heatwave Number)
- Core idea: number of heatwaves

## HWF (Heatwave Frequency)
- Core idea: number of heatwave days

## HWD (Heatwave Duration)
- Core idea: longest heatwave

## HWM (Heatwave Magnitude)
- Core idea: cumulative intensity

## HWA (Heatwave Amplitude)
- Core idea: peak intensity

- Definition:
  - ≥3 consecutive days exceeding thresholds (EHF or percentile)
- Interpretation: comprehensive heatwave characterization

---

# 14. Coldwave Indices (ECF-based)

## CWN, CWF, CWD, CWM, CWA
- Core idea: coldwave analogues of heatwave metrics
- Method: Excess Cold Factor (ECF)
- Interpretation: cold extreme events

---

# Data Requirements (All Indices)

Minimum:
- Daily time series of:
  - TX (max temp)
  - TN (min temp)
  - TG/TM (mean temp)
  - RR (precipitation)

Optional:
- Reference climatology (for percentile indices)
- Base thresholds (for degree days)

---

# Climdex Indices Mapped to Use-Cases

---

# 1. Energy Systems (Heating / Cooling Demand)

## Core objective
Quantify **temperature-driven energy demand**, peak load risk, and seasonality shifts.

## Primary indices

### Heating demand
- HDDheatn
  - Direct proxy for heating energy consumption
  - Sensitive to cold-season warming trends

- FD (Frost Days)
- ID (Icing Days)
  - Indicate extreme cold stress on infrastructure

### Cooling demand
- CDDcoldn
  - Direct proxy for cooling demand

- SU (Summer Days)
- TR (Tropical Nights)
  - Capture sustained heat affecting cooling loads and human comfort

## Peak load / extremes
- TXx, TNx
  - Grid stress during extreme heat events

- WSDI
  - Prolonged cooling demand spikes

## Interpretation
- HDD ↓ + CDD ↑ → electrification pressure shifts
- TR ↑ → nighttime cooling demand rises (critical for grids)

---

# 2. Hydrology & Water Resources

## Core objective
Assess **water availability, flood risk, drought persistence, and precipitation structure**.

## Precipitation totals & intensity
- PRCPTOT → total water input
- SDII → rainfall intensity (runoff vs infiltration)

## Flood risk / extremes
- Rx1day → flash floods
- Rx5day → basin-scale flooding
- R95p / R99p → extreme precipitation contribution

## Event frequency
- R10mm / R20mm → heavy rainfall frequency

## Persistence
- CWD → prolonged wet periods (soil saturation, flood risk)
- CDD → drought duration

## Drought metrics
- SPI → meteorological drought
- SPEI → hydroclimatic drought (includes evapotranspiration)

## Interpretation
- ↑ SDII + ↑ Rx1day → more intense storms
- ↑ CDD → longer dry spells (agriculture stress)
- ↑ R95pTOT → rainfall concentrated in extremes

---

# 3. Climate Extremes Detection

## Core objective
Quantify **frequency, intensity, and persistence of extreme events**

## Temperature extremes
- TXx, TNn → absolute extremes
- TX90p, TN90p → hot extremes frequency
- TX10p, TN10p → cold extremes frequency

## Event persistence
- WSDI → heatwaves
- CSDI → cold spells

## Compound extremes
- TXbdTNbd → joint cold day/night extremes

## Range-based
- ETR (TXx − TNn) → extreme spread

## Interpretation
- TX90p ↑ + TN90p ↑ → systematic warming of extremes
- WSDI ↑ → longer/more frequent heatwaves
- TN10p ↓ → fewer cold extremes

---

# 4. Agriculture & Ecosystems

## Core objective
Evaluate **growing conditions, stress thresholds, and phenology**

## Growing conditions
- GSL (Growing Season Length)
- GDDgrown (Growing Degree Days)

## Stress thresholds
- TXge30 / TXge35 → heat stress
- TNlt0 / FD → frost damage risk

## Water stress
- CDD → drought stress
- SPI / SPEI → moisture deficit

## Variability
- DTR → plant stress, evapotranspiration dynamics

## Interpretation
- GSL ↑ + GDD ↑ → longer but potentially heat-stressed seasons
- ↑ TXge35 → crop failure risk
- ↓ FD → reduced frost risk (but possible phenology mismatch)

---

# 5. Urban Climate & Human Health

## Core objective
Assess **heat stress, mortality risk, and livability**

## Heat stress indicators
- TR (Tropical Nights) → lack of nighttime relief
- TX90p / TN90p → extreme heat frequency
- WSDI → heatwave persistence

## Cold stress
- TN10p, CSDI

## Composite heatwave metrics
- HWF (frequency)
- HWD (duration)
- HWA (amplitude)

## Interpretation
- TR ↑ is strongly linked to mortality risk
- WSDI ↑ → prolonged exposure events
- Nighttime warming (TN90p ↑) is critical for health

---

# 6. Infrastructure & Risk Engineering

## Core objective
Quantify **design loads, failure risk, and resilience requirements**

## Thermal stress
- TXx / TNn → design extremes
- DTR → material fatigue

## Freeze–thaw cycles
- FD, ID

## Flooding / drainage
- Rx1day, Rx5day
- SDII

## Persistence risks
- CWD → prolonged wet stress
- CDD → subsidence / soil shrinkage

## Interpretation
- ↑ Rx1day → drainage system overload
- ↑ DTR → expansion/contraction stress
- ↓ FD but ↑ variability → more freeze–thaw cycling risk

---

# 7. Climate Change Detection & Attribution

## Core objective
Identify **systematic shifts in climate distributions**

## Most sensitive indices
- TX90p, TN90p (warming signal)
- TN10p, TX10p (cold tail shrinkage)
- WSDI / CSDI (persistence changes)

## Supporting indicators
- DTR → changes in radiative/cloud regimes
- PRCPTOT vs R95p → redistribution of rainfall

## Interpretation
- Asymmetric warming:
  - TN90p ↑ faster than TX90p → nighttime amplification
- Extremes intensify faster than means

---

# 8. Compound & Multivariate Risk Analysis

## Core objective
Capture **interacting hazards**

## Typical combinations
- Heat + drought:
  - TX90p + CDD + SPEI

- Flood risk:
  - Rx5day + CWD + antecedent PRCPTOT

- Energy stress:
  - WSDI + TR + CDDcoldn

## Interpretation
- Risk is often nonlinear when indices co-occur
- Compound indices outperform single metrics in impact studies

---

# Quick Index Selection Guide

| Use-case              | Key indices |
|----------------------|------------|
| Heating demand       | HDD, FD, ID |
| Cooling demand       | CDD, SU, TR |
| Flood risk           | Rx1day, Rx5day, R95p |
| Drought              | CDD, SPI, SPEI |
| Heatwaves            | WSDI, TX90p |
| Agriculture          | GSL, GDD, TXge30 |
| Climate change       | TN90p, TX90p, WSDI |
| Infrastructure       | TXx, Rx1day, DTR |

---

# Practical Notes

- Prefer **percentile indices** for climate change studies (robust to location)
- Use **absolute thresholds** for impact studies (engineering, agriculture)
- Combine:
  - intensity (e.g., TXx)
  - frequency (TX90p)
  - duration (WSDI)
  for complete risk characterization

---

---

# Resources

- Climdex index definitions:
  https://www.climdex.org/learn/indices/
- Background:
  - ETCCDI indices (WMO standard)
  - Zhang et al. (2005) – percentile methodology
- Tools:
  - climdex.pcic (R)
  - CDO / icclim libraries
- Data:
  - station observations
  - reanalysis (ERA5)
  - climate model output

---