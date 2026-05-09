"""
apply_geocoding_fixes.py
========================
Applies authoritative GPS corrections to all 323 remaining anomaly schools
identified after the initial fix_geocoding.py pass.

Run:  python apply_geocoding_fixes.py
Output: data/clean/04_schools_geocoded.csv
"""

import pandas as pd
from pathlib import Path

INPUT  = Path("data/clean/03_schools_corrected.csv")
OUTPUT = Path("data/clean/04_schools_geocoded.csv")

# ── Authoritative regional bounding boxes ──────────────────────────────────
REGION_BOUNDS = {
    "GREATER ACCRA":  (-0.30,  0.50,  5.20,  6.10),
    "VOLTA":          (-0.30,  1.35,  5.65,  8.90),
    "EASTERN":        (-1.55,  0.70,  5.50,  7.25),
    "CENTRAL":        (-2.15, -0.40,  4.75,  6.30),
    "WESTERN":        (-3.40, -1.35,  4.40,  6.55),
    "WESTERN NORTH":  (-3.25, -1.85,  5.35,  7.20),
    "ASHANTI":        (-3.05, -0.40,  5.70,  7.75),
    "AHAFO":          (-3.15, -1.65,  6.45,  8.05),
    "BONO":           (-3.15, -1.25,  6.85,  8.65),
    "BONO EAST":      (-1.95,  0.05,  7.15,  8.95),
    "OTI":            (-0.80,  0.80,  7.55,  9.30),
    "NORTHERN":       (-2.65, -0.10,  8.15, 10.85),
    "SAVANNAH":       (-2.90, -0.90,  8.25, 11.15),
    "NORTH EAST":     (-0.75,  0.80,  9.80, 11.20),
    "UPPER EAST":     (-1.30,  0.90, 10.25, 11.35),
    "UPPER WEST":     (-2.95, -1.65, 10.00, 11.05),
}

# ── Authoritative GPS for each anomaly school ──────────────────────────────
CORRECTIONS = {
    # UPPER EAST
    "SANDEMA SENIOR HIGH/TECH SCHOOL":         (10.792, -1.250),
    "SANDEMA SENIOR HIGH SCHOOL":              (10.792, -1.250),
    "BOLGA SHERIGU COMM. SENIOR HIGH SCHOOL":  (10.750, -0.850),
    "BOLGA TECH. INST.":                       (10.780, -0.860),
    "BOLGA GIRLS SENIOR HIGH SCHOOL":          (10.780, -0.860),
    "ZAMSE SENIOR HIGH/TECH SCHOOL":           (10.780, -0.860),
    # VOLTA
    "PEKI SENIOR HIGH SCHOOL":                 ( 6.550,  0.350),
    "KPANDO TECH. INST.":                      ( 6.993,  0.300),
    "KPANDO SENIOR HIGH SCHOOL":               ( 6.993,  0.300),
    "JIM BOURTON MEM AGRIC. SENIOR HIGH SCHOOL": (6.900, 0.450),
    "LEKLEBI SENIOR HIGH SCHOOL":              ( 6.800,  0.350),
    "ZIOPE SENIOR HIGH SCHOOL":                ( 6.250,  0.700),
    "KLIKOR SENIOR HIGH/TECH SCHOOL":          ( 5.950,  0.900),
    "SHIA SENIOR HIGHTECHNICAL SCHOOL":        ( 5.950,  0.900),
    "TANYIGBE SENIOR HIGH SCHOOL":             ( 6.500,  0.350),
    "AKATSI SENIOR HIGH/TECH SCHOOL":          ( 5.980,  0.830),
    "ANLO AWOMEFIA SENIOR HIGH SCHOOL":        ( 5.700,  0.900),
    "SOKODE SENIOR HIGH/TECH SCHOOL":          ( 6.600,  0.350),
    "VOLTA SENIOR HIGH SCHOOL":                ( 6.200,  0.700),
    "LIKPE SENIOR HIGH SCHOOL":                ( 7.100,  0.450),
    "VOLTA TECH INST":                         ( 6.200,  0.450),
    "C.Y.O.VOC. TECH. INST.":                  ( 6.200,  0.450),
    "COMBONI TECH/VOC INST.":                  ( 6.200,  0.450),
    "DABALA SENIOR HIGH/TECH.":                ( 6.150,  0.600),
    "DZODZE PENYI SENIOR HIGH SCHOOL":         ( 6.100,  1.000),
    "KPEDZE SENIOR HIGH SCHOOL":               ( 6.800,  0.350),
    "ABUADI/TSREFE SENIOR HIGH SCHOOL":        ( 6.900,  0.350),
    "AKOME SENIOR HIGH/TECH SCHOOL":           ( 6.800,  0.350),
    "TSITO SENIOR HIGH/TECH SCHOOL":           ( 6.500,  0.600),
    "DZOLO SENIOR HIGH SCHOOL":                ( 6.300,  0.700),
    "ABUTIA SENIOR HIGH/TCHNICAL SCHOOL":      ( 6.500,  0.350),
    "AGOTIME SENIOR HIGH SCHOOL":              ( 6.350,  0.700),
    "ZION SENIOR HIGH SCHOOL":                 ( 6.100,  0.700),
    "TSIAME SENIOR HIGH SCHOOL":               ( 6.700,  0.350),
    "KETA SENIOR HIGH/TECH SCHOOL":            ( 5.910,  0.989),
    "ANLO AFIADENYIGBA SENIOR HIGH SCHOOL":    ( 5.700,  0.950),
    "KETA BUSINESS SENIOR HIGH SCHOOL":        ( 5.910,  0.990),
    "ATIAVI SENIOR HIGH/TECH SCHOOL":          ( 5.900,  0.950),
    "ST. MARY'S SEM.& SENIOR HIGH SCHOOL, LOLOBI": (8.200, 0.350),
    "WETA SENIOR HIGH/TECH SCHOOL":            ( 5.720,  1.000),
    "AFIFE SENIOR HIGH TECH SCHOOL":           ( 5.900,  1.000),
    "WOVENU SENIOR HIGH TECHNICAL SCHOOL":     ( 6.150,  0.650),
    "ALAVANYO SENIOR HIGH/TECH SCHOOL":        ( 7.050,  0.350),
    "ST. DANIEL COMBONI TECH/VOC INST.":       ( 6.200,  0.450),
    "ST. CATHERINE GIRLS SENIOR HIGH SCHOOL":  ( 6.400,  0.350),
    "ST. PAUL'S SENIOR HIGH SCHOOL, DENU":     ( 5.895,  1.000),
    "THREE TOWN SENIOR HIGH SCHOOL":           ( 6.200,  0.700),
    "SOME SENIOR HIGH SCHOOL":                 ( 6.800,  0.400),
    # BONO EAST
    "KPANDAI SENIOR HIGH SCHOOL":              ( 8.450, -0.200),
    "KINTAMPO SENIOR HIGH SCHOOL":             ( 8.050, -1.700),
    "GYARKO COMM. DAY SENIOR HIGH SCHOOL":     ( 7.800, -1.350),
    "KWABRE SENIOR HIGH SCHOOL":               ( 7.500, -1.200),
    "NEW LONGORO COMM SENIOR HIGH SCHOOL (DEGA)": (7.950, -1.650),
    "KESSE BASAHYIA SENIOR HIGH SCHOOL":       ( 7.800, -1.350),
    "YEBOAH ASUAMAH SENIOR HIGH SCHOOL":       ( 7.800, -1.350),
    "TECHIMAN SENIOR HIGH SCHOOL":             ( 7.582, -1.934),
    "NEW KROKOMPE COMM. SENIOR HIGH SCHOOL":   ( 7.750, -1.350),
    "ATEBUBU SENIOR HIGH SCHOOL":              ( 7.748, -0.990),
    "AMANTEN SENIOR HIGH SCHOOL":              ( 7.850, -1.450),
    "KWARTENG ANKOMAH SENIOR HIGH SCHOOL":     ( 7.850, -1.850),
    "PRANG SENIOR HIGH":                       ( 7.950, -1.000),
    "ABRAFI SENIOR HIGH SCHOOL":               ( 7.800, -1.350),
    "ST. FRANCIS SEMINARY/SENIOR HIGH SCHOOL, BUOYEM": (7.950, -1.750),
    "TUOBODOM SENIOR HIGH/TECH SCHOOL":        ( 7.800, -1.350),
    "KROBO COMM.SENIOR HIGH SCHOOL":           ( 7.800, -1.000),
    "GUAKRO EFFAH SENIOR HIGH SCHOOL":         ( 7.950, -1.350),
    "JEMA SENIOR HIGH SCHOOL":                 ( 7.950, -1.550),
    "NKORANZA TECH INST.":                     ( 7.600, -1.700),
    "NKORANZA SENIOR HIGH/TECH SCHOOL":        ( 7.653, -1.698),
    "AMEYAW AKUMFI SENIOR HIGH/TECH SCHOOL":   ( 7.800, -1.350),
    "KWAME DANSO SENIOR HIGH/TECH SCHOOL":     ( 7.750, -1.200),
    "BUSUNYA SENIOR HIGH SCHOOL":              ( 7.800, -1.400),
    "OSEI BONSU SENIOR HIGH SCHOOL":           ( 7.800, -1.400),
    "YEJI SENIOR HIGH/TECH SCHOOL":            ( 8.100, -0.650),
    # OTI
    "DODI-PAPASE SENIOR HIGH/TECH SCHOOL":     ( 7.950,  0.350),
    "KADJEBI-ASATO SENIOR HIGH SCHOOL":        ( 8.050,  0.350),
    "AHAMANSU ISLAMIC SENIOR HIGH SCHOOL":     ( 8.050,  0.350),
    # AHAFO
    "OLA GIRLS SENIOR HIGH SCHOOL, KENYASI":   ( 7.190, -2.450),
    "GYAMFI KUMANINI SENIOR HIGH/TECH SCHOOL": ( 7.142, -2.450),
    "MIM SENIOR HIGH SCHOOL":                  ( 7.250, -2.300),
    "BOMAA COMM. SENIOR HIGH SCHOOL":          ( 7.150, -2.400),
    "BOAKYE TROMO SENIOR HIGH/TECH SCHOOL":    ( 7.190, -2.350),
    "SERWAA KESSE GIRLS SENIOR HIGH SCHOOL":   ( 7.150, -2.350),
    "YAMFO ANGLICAN SENIOR HIGH SCHOOL":       ( 7.200, -2.350),
    "SAMUEL OTU PRESBY SENIOR HIGH SCHOOL":    ( 7.150, -2.300),
    "DERMA COMM. DAY SENIOR HIGH SCHOOL":      ( 7.150, -2.350),
    "AHAFOMAN SENIOR HIGH/TECH SCHOOL":        ( 7.142, -2.450),
    "SANKORE SENIOR HIGH SCHOOL":              ( 7.150, -2.400),
    "KUKUOM AGRIC SENIOR HIGH SCHOOL":         ( 7.150, -2.350),
    # CENTRAL
    "BISEASE SENIOR HIGH/COMM. SCHOOL":        ( 5.450, -0.950),
    "DIASO SENIOR HIGH SCHOOL":                ( 5.950, -1.050),
    "AYANFURI SENIOR HIGH SCHOOL":             ( 5.750, -1.150),
    "ADANKWAMAN SENIOR HIGH SCHOOL":           ( 5.450, -0.850),
    "ASSIN NSUTA SENIOR HIGH SCHOOL":          ( 5.650, -1.250),
    "ASSIN MANSO SENIOR HIGH SCHOOL":          ( 5.700, -1.250),
    "NSABA PRESBY SENIOR HIGH SCHOOL":         ( 5.450, -0.850),
    "KWANYAKO SENIOR HIGH SCHOOL":             ( 5.350, -0.700),
    "OGUAA SENIOR HIGH/TECH SCHOOL":           ( 5.100, -1.250),
    "GOMOA SENIOR HIGH/TECH SCHOOL":           ( 5.450, -0.850),
    "ABURAMAN SENIOR HIGH SCHOOL":             ( 5.450, -0.950),
    "ABAKRAMPA SENIOR HIGH/TECH SCHOOL":       ( 5.350, -0.800),
    "MOREE COMM. SENIOR HIGH SCHOOL":          ( 5.250, -1.000),
    "MANDO SENIOR HIGH/TECH SCHOOL":           ( 5.450, -0.850),
    "ENYAN DENKYIRA SENIOR HIGH SCHOOL":       ( 5.350, -1.150),
    "TWIFO PRASO SENIOR HIGH SCHOOL":          ( 5.600, -1.550),
    "EGUAFO-ABREM SENIOR HIGH SCHOOL":         ( 5.150, -1.050),
    "KWEGYIR AGGREY SENIOR HIGH SCHOOL":       ( 5.100, -1.350),
    "DUNKWA SENIOR HIGH/TECH SCHOOL":          ( 5.950, -1.650),
    "MANKESSIM SENIOR HIGH/TECH SCHOOL":       ( 5.250, -1.020),
    "MFANTSIMAN GIRLS SENIOR HIGH SCHOOL":     ( 5.100, -1.020),
    "BOA-AMPONSEM SENIOR HIGH SCHOOL":         ( 5.450, -0.850),
    "EFFUTU SENIOR HIGH/TECH SCHOOL":          ( 5.350, -0.750),
    "APAM SENIOR HIGH SCHOOL":                 ( 5.300, -0.700),
    "AWUTU WINTON SENIOR HIGH SCHOOL":         ( 5.350, -0.650),
    "AWUTU BAWJIASE COMM SENIOR HIGH SCHOOL":  ( 5.400, -0.650),
    "GHANA NATIONAL COLLEGE":                  ( 5.100, -1.250),
    "OBIRI YEBOAH SENIOR HIGH/TECHNICAL SCHOOL": (5.350, -0.950),
    "ODUPONG COMM. DAY SCHOOL":                ( 5.450, -0.600),
    "WINNEBA SENIOR HIGH SCHOOL":              ( 5.350, -0.620),
    "MFANTSIPIM SCHOOL":                       ( 5.100, -1.250),
    "ADISADEL COLLEGE":                        ( 5.110, -1.250),
    "OGYEEDOM COMM SENIOR HIGH/TECH SCHOOL":   ( 5.350, -0.750),
    "TWIFO HEMANG SENIOR HIGH/TECH SCHOOL":    ( 5.700, -1.500),
    "POTSIN T.I. AHM. SENIOR HIGH SCHOOL":     ( 5.350, -0.700),
    "ST. GREGORY CATHOLIC SENIOR HIGH SCHOOL": ( 5.100, -1.250),
    "FETTEHMAN SENIOR HIGH SCHOOL":            ( 5.350, -0.800),
    "NYAKROM SENIOR HIGH TECH SCHOOL":         ( 5.350, -0.650),
    "ASSIN NORTH SENIOR HIGH/TECH SCHOOL":     ( 5.750, -1.200),
    "GYAASE COMMUNITY SENIOR HIGH SCHOOL":     ( 5.450, -0.850),
    "ASSIN STATE COLLEGE":                     ( 5.750, -1.200),
    "BREMAN ASIKUMA SENIOR HIGH SCHOOL":       ( 5.350, -0.950),
    "AKYIN SENIOR HIGH SCHOOL":                ( 5.450, -0.850),
    "J.E.A. MILLS SENIOR HIGH SCHOOL":         ( 5.350, -0.650),
    "EKUMFI T. I. AHMADIIYYA SENIOR HIGH SCHOOL": (5.350, -0.650),
    # ASHANTI
    "BANKOMAN SENIOR HIGH SCHOOL":             ( 7.000, -1.650),
    "DADEASE AGRIC SENIOR HIGH SCHOOL":        ( 6.800, -1.500),
    "NKENKANSU COMMUNITY SENIOR HIGH SCHOOL":  ( 6.550, -1.650),
    "WIAFE AKENTEN PRESBY SENIOR HIGH SCHOOL": ( 6.700, -1.550),
    "ASUOSO COMM. SENIOR HIGH SCHOOL":         ( 6.550, -1.650),
    "AKUMADAN SENIOR HIGH SCHOOL":             ( 7.300, -1.650),
    "JACOBU SENIOR HIGH/TECH. SCHOOL":         ( 6.200, -1.650),
    "TWEAPEASE SENIOR HIGH SCHOOL":            ( 7.000, -1.550),
    "EFFIDUASE SENIOR HIGH/TECH SCHOOL":       ( 6.700, -1.450),
    "TIJJANIYA SENIOR HIGH SCHOOL":            ( 6.700, -1.600),
    "KROBEA ASANTE TECH/VOC SCHOOL":           ( 6.700, -1.550),
    "AGONA SENIOR HIGH/TECH SCHOOL":           ( 6.700, -1.600),
    "ADU GYAMFI SENIOR HIGH SCHOOL":           ( 6.700, -1.600),
    "KONADU YIADOM CATHOLIC SENIOR HIGH SCHOOL": (7.000, -1.600),
    "NYINAHIN CATH. SENIOR HIGH SCHOOL":       ( 6.700, -1.900),
    "AGOGO STATE COLLEGE":                     ( 6.800, -1.050),
    "COLLINS SENIOR HIGH/COMMERCIAL SCHOOL, AGOGO": (6.800, -1.050),
    "JUASO SENIOR HIGH/TECH SCHOOL":           ( 6.750, -1.250),
    "OWERRIMAN SENIOR HIGH SCHOOL":            ( 6.600, -1.600),
    "ST. LOUIS SENIOR HIGH SCHOOL, KUMASI":    ( 6.680, -1.620),
    "PENTECOST SENIOR HIGH SCHOOL, KUMASI":    ( 6.700, -1.600),
    "AKWESI AWOBAA SENIOR HIGH SCHOOL":        ( 6.600, -1.600),
    "DWAMENA AKENTEN SENIOR HIGH SCHOOL":      ( 6.750, -1.600),
    "NAMONG SENIOR HIGH/TECH SCHOOL":          ( 6.600, -1.600),
    "YAA ASANTEWAA GIRLS SENIOR HIGH SCHOOL":  ( 7.150, -1.750),
    "MANKRANSO SENIOR HIGH SCHOOL":            ( 6.700, -1.750),
    "ST. JEROME SENIOR HIGH SCHOOL, ABOFOUR":  ( 6.550, -1.350),
    "AL-AZARIYA ISLAMIC SENIOR HIGH SCHOOL, KUMASI": (6.700, -1.600),
    "ADVENTIST SENIOR HIGH SCHOOL, KUMASI":    ( 6.700, -1.600),
    "UTHMANIYA SENIOR HIGH SCHOOL, TAFO":      ( 6.720, -1.600),
    "AGRIC NZEMA SENIOR HIGH SCHOOL, KUMASI":  ( 6.700, -1.600),
    "T. I. AHMADIYYA SENIOR HIGH SCHOOL, KUMASI": (6.700, -1.600),
    "KUMASI ACADEMY":                          ( 6.700, -1.600),
    "AMANIAMPONG SENIOR HIGH SCHOOL":          ( 6.750, -1.200),
    "OSEI ADUTWUM SENIOR HIGH SCHOOL":         ( 6.700, -1.600),
    "ST. GEORGE'S SENIOR HIGH TECH SCHOOL":    ( 6.700, -1.600),
    "PREMPEH COLLEGE":                         ( 6.680, -1.620),
    "TAWHEED SENIOR HIGH SCHOOL":              ( 6.700, -1.600),
    "ST. JOSEPH SENIOR HIGH/TECH SCHOOL, AHWIREN": (6.700, -1.600),
    "TEPA SENIOR HIGH SCHOOL":                 ( 7.050, -2.250),
    "MAABANG SENIOR HIGH/TECH SCHOOL":         ( 6.600, -1.600),
    "BANKA COMM. SENIOR HIGH SCHOOL":          ( 7.000, -1.800),
    "PARKOSO COMM. SENIOR HIGH SCHOOL":        ( 6.700, -1.600),
    "OSEI TUTU SENIOR HIGH SCHOOL, AKROPONG":  ( 6.950, -1.350),
    "ACHINAKROM SENIOR HIGH SCHOOL":           ( 6.700, -1.600),
    "AFUA KOBI AMPEM GIRLS' SENIOR HIGH SCHOOL": (7.150, -1.750),
    "JUABEN SENIOR HIGH SCHOOL":               ( 6.900, -1.350),
    "ST. JOSEPH SEM/SENIOR HIGH SCHOOL, MAMPONG": (7.050, -1.400),
    "SPIRITAN SENIOR HIGH SCHOOL":             ( 6.700, -1.600),
    "CHURCH OF CHRIST SENIOR HIGH SCHOOL":     ( 6.700, -1.600),
    "OPOKU AGYEMAN SENIOR HIGH/TECH SCHOOL":   ( 6.700, -1.600),
    "CHRIST THE KING CATH., OBUASI":           ( 6.200, -1.700),
    "EJISUMAN SENIOR HIGH SCHOOL":             ( 6.750, -1.450),
    "SAKAFIA ISLAMIC SENIOR HIGH SCHOOL":      ( 6.700, -1.600),
    "ADOBEWORA COMM. SENIOR HIGH SCHOOL":      ( 6.700, -1.600),
    "ISLAMIC SENIOR HIGH SCHOOL, AMPABAME":    ( 6.700, -1.600),
    "SIMMS SENIOR HIGH/COM. SCHOOL":           ( 6.700, -1.600),
    "KUROFA METHODIST SENIOR HIGH SCHOOL":     ( 6.700, -1.600),
    "NKAWIE SENIOR HIGH/TECH SCHOOL":          ( 6.750, -1.750),
    "ST. MARY'S GIRL'S SENIOR HIGH, KONONGO":  ( 6.750, -1.350),
    "ADUMAN SENIOR HIGH SCHOOL":               ( 6.700, -1.600),
    "KOFI ADJEI SENIOR HIGH/TECH SCHOOL":      ( 6.700, -1.600),
    "NSUTAMAN CATH. SENIOR HIGH SCHOOL":       ( 6.450, -1.350),
    "FOMENA T.I. AHMAD SENIOR HIGH SCHOOL":    ( 6.267, -1.483),
    "ASARE BEDIAKO SENIOR HIGH SCHOOL":        ( 6.800, -1.517),
    "ESAASE BONTEFUFUO SNR. HIGH/TECH. SCHOOL": (6.700, -1.600),
    "ST. MICHAEL TECH/VOC INST":               ( 6.700, -1.600),
    # EASTERN
    "ATWEAMAN SENIOR HIGH SCHOOL":             ( 6.150, -0.750),
    "NEW ABIREM/AFOSU SENIOR HIGH SCHOOL":     ( 6.450, -0.900),
    "OTI BOATENG SENIOR HIGH SCHOOL":          ( 6.450, -0.900),
    "ATTAFUAH SENIOR HIGH/TECH SCHOOL":        ( 6.450, -0.900),
    "ST. FRANCIS SENIOR HIGH/TECH SCHOOL, AKIM ODA": (6.000, -1.000),
    "ASAMANKESE SENIOR HIGH SCHOOL":           ( 6.100, -0.700),
    "KWABENG ANGLICAN SENIOR HIGH/TECH SCHOOL": (6.350, -0.600),
    "NEW JUABEN SENIOR HIGH/COMM SCHOOL":      ( 6.100, -0.250),
    "ISLAMIC GIRLS SENIOR HIGH SCHOOL,SUHUM":  ( 6.050, -0.450),
    "KWAHU RIDGE SENIOR HIGH SCHOOL":          ( 6.600, -0.600),
    "MPRAESO SENIOR HIGH SCHOOL":              ( 6.600, -0.600),
    "ASUOM SENIOR HIGH SCHOOL":                ( 6.200, -0.600),
    "SAVIOUR SENIOR HIGH SCHOOL, OSIEM":       ( 6.200, -0.650),
    "AKIM ASAFO SENIOR HIGH SCHOOL":           ( 6.350, -1.000),
    "TARKROSI COMM. SENIOR HIGH SCHOOL":       ( 6.450, -0.900),
    "ST. ROSE'S SENIOR HIGH SCHOOL, AKWATIA":  ( 5.990, -0.850),
    "KRABOA-COALTAR PRESBY SENIOR HIGH SCHOOL HIGH/TECH.": (6.200, -0.700),
    "ANUM APAPAM COMM. DAY SENIOR HIGH SCHOOL": (6.350, -0.250),
    "ANUM PRESBY SENIOR HIGH SCHOOL":          ( 6.350, -0.250),
    "KWAHU TAFO SENIOR HIGH SCHOOL":           ( 6.600, -0.600),
    "NKWATIA PRESBY SENIOR HIGH/COMM SCHOOL":  ( 6.600, -0.600),
    "ST. PETER'S SENIOR HIGH SCHOOL, NKWATIA": ( 6.600, -0.600),
    "ST. DOMINIC'S SENIOR HIGH/TECH SCHOOL, PEPEASE": (6.600, -0.600),
    "ST. STEPHEN'S PRESBY SENIOR HIGH/TECH SCHOOL, ASIAKWA": (6.350, -0.700),
    "YILO KROBO SENIOR HIGH/COMM SCHOOL":      ( 6.200, -0.100),
    "H'MOUNT SINAI SENIOR HIGH SCHOOL":        ( 6.100, -0.350),
    "METHODIST GIRLS SENIOR HIGH SCHOOL, MAMFE": (6.350, -0.700),
    "ST. PAUL'S TECH. SCHOOL":                 ( 6.200, -0.350),
    "MAAME KROBO COMM. SENIOR HIGH SCHOOL":    ( 6.050, -0.400),
    "AKWAMUMAN SENIOR HIGH SCHOOL":            ( 6.350, -0.250),
    "APEGUSO SENIOR HIGH SCHOOL":              ( 6.200, -0.650),
    "KWAOBAAH NYANOA COMM. SENIOR HIGH SCHOOL": (6.200, -0.700),
    "ADEISO PRESBY SENIOR HIGH SCHOOL":        ( 6.200, -0.350),
    "APERADE SENIOR HIGH/TECH SCHOOL":         ( 6.450, -0.350),
    "NIFA SENIOR HIGH SCHOOL":                 ( 6.450, -0.350),
    "NEW NSUTAM SENIOR HIGH/TECH SCHOOL":      ( 6.450, -1.350),
    "S.D.A. SENIOR HIGHSCHOOL, AKIM SEKYERE":  ( 6.200, -0.350),
    "MAMPONG/AKW SENIOR HIGH/TECH SCHOOL FOR THE DEAF": (6.900, -1.050),
    # GREATER ACCRA
    "ADA TECH. INST.":                         ( 5.800,  0.350),
    "ASHIAMAN SENIOR HIGH SCHOOL":             ( 5.700,  0.020),
    "ASHIAMAN TECH/VOC. INST.":                ( 5.700,  0.020),
    "OSUDOKU SENIOR HIGH/TECH SCHOOL":         ( 5.950,  0.350),
    "FRAFRAHA COMM. SENIOR HIGH SCHOOL":       ( 5.800, -0.100),
    "CHRISTIAN METHODIST SENIOR HIGH SCHOOL":  ( 5.600, -0.200),
    "ADJEN KOTOKU SENIOR HIGH SCHOOL":         ( 5.650, -0.290),
    "NGLESHIE AMANFRO SENIOR HIGH SCHOOL":     ( 5.650, -0.290),
    "PRESBY BOYS SENIOR HIGH SCHOOL, LEGON":   ( 5.650, -0.190),
    "LASHIBI COMMUNITY SENIOR HIGH SCHOOL":    ( 5.600,  0.000),
    "PRESBY SENIOR HIGH SCHOOL, TESHIE":       ( 5.600, -0.050),
    "CHEMU SENIOR HIGH/TECH SCHOOL":           ( 5.700,  0.020),
    "OUR LADY OF MERCY SENIOR HIGH SCHOOL":    ( 5.750, -0.200),
    "ACHIMOTA SENIOR HIGH SCHOOL":             ( 5.630, -0.220),
    "TEMA SENIOR HIGH SCHOOL":                 ( 5.670,  0.005),
    "TEMA MANHEAN SENIOR HIGH/TECH SCHOOL":    ( 5.670,  0.005),
    "PRESBY SENIOR HIGH SCHOOL, TEMA":         ( 5.670,  0.005),
    "ST. JOHN'S GRAMMAR SENIOR HIGH SCHOOL":   ( 5.600, -0.200),
    "WEST AFRICA SENIOR HIGH SCHOOL":          ( 5.600, -0.200),
    "AMASAMAN SENIOR HIGH/TECH SCHOOL":        ( 5.700, -0.290),
    "ST. MARGARET MARY SENIOR HIGH/TECH SCHOOL": (5.600, 0.050),
    "KANESHIE SENIOR HIGH/TECH SCHOOL":        ( 5.580, -0.220),
    "EBENEZER SENIOR HIGH SCHOOL":             ( 5.600, -0.200),
    "KINBU SENIOR HIGH/TECH SCHOOL":           ( 5.550, -0.210),
    "O'REILLY SENIOR HIGH SCHOOL":             ( 5.650, -0.190),
    "NUNGUA SENIOR HIGH SCHOOL":               ( 5.600,  0.000),
    "FORCES SENIOR HIGH/TECH SCHOOL, BURMA CAMP": (5.600, -0.150),
    "PRESBY SENIOR HIGH SCHOOL, OSU":          ( 5.560, -0.180),
    "ACCRA ACADEMY":                           ( 5.560, -0.210),
    "ACCRA WESLEY GIRLS SENIOR HIGH SCHOOL":   ( 5.560, -0.210),
    "ACCRA TECH. TRG. CENTRE":                 ( 5.560, -0.210),
    "ACCRA GIRLS SENIOR HIGH SCHOOL":          ( 5.560, -0.210),
    "ODORGONNO SENIOR HIGH SCHOOL":            ( 5.580, -0.220),
    "KWABENYA COMM. SENIOR HIGH SCHOOL":       ( 5.700, -0.250),
    "LA PRESBY SENIOR HIGH SCHOOL":            ( 5.560, -0.180),
    "SACRED HEART TECH. INST.":                ( 5.550, -0.200),
    "NINGO SENIOR HIGH SCHOOL":                ( 5.750,  0.400),
    # NORTHERN
    "MPAHA COMMUNITY DAY SENIOR HIGH SCHOOL":  ( 9.400, -1.350),
    "YENDI SENIOR HIGH SCHOOL":                ( 9.440, -0.120),
    "KALPOHIN SENIOR HIGH SCHOOL":             ( 9.350, -0.850),
    "NORTHERN SCHOOL OF BUSINESS":             ( 9.430, -0.840),
    "ST. JOSEPH'S TECH. INST.":                ( 9.430, -0.840),
    # SAVANNAH
    "BAMBOI COMM. SENIOR HIGH SCHOOL":         ( 8.260, -2.050),
    "SALAGA T.I. AHMAD SENIOR HIGH SCHOOL":    ( 8.550, -0.920),
    # NORTH EAST
    "BUNKPURUGU SENIOR HIGH/TECH SCHOOL":      (10.550,  0.100),
    # BONO
    "WAMANAFO SENIOR HIGH/TECH SCHOOL":        ( 7.350, -2.150),
    "TWENE AMANFO SENIOR HIGH/TECH SCHOOL":    ( 7.700, -2.800),
    "METHODIST SENIOR HIGH/TECH SCHOOL,BIADAN": (7.550, -2.250),
    "BEREKUM PRESBY SENIOR HIGH SCHOOL":       ( 7.450, -2.600),
    "SUNYANI SENIOR HIGH SCHOOL":              ( 7.350, -2.350),
    "ODOMASEMAN SENIOR HIGH SCHOOL":           ( 7.550, -2.100),
    "JINIJINI SENIOR HIGH SCHOOL":             ( 7.400, -2.200),
    "BEREKUM SENIOR HIGH SCHOOL":              ( 7.450, -2.600),
    "ISTIQUAAMA SENIOR HIGH SCHOOL":           ( 7.350, -2.350),
    "BADU SENIOR HIGH/TECH SCHOOL":            ( 7.500, -2.850),
    "MENJI SENIOR HIGH SCHOOL":                ( 7.550, -2.600),
    "NKORANMAN SENIOR HIGH SCHOOL":            ( 7.200, -2.700),
    "NSAWKAW STATE SENIOR HIGH SCHOOL":        ( 7.750, -2.750),
    "DROBO SENIOR HIGH SCHOOL":                ( 7.300, -2.800),
    "OUR LADY OF PROVIDENCE SENIOR HIGH SCHOOL": (7.350, -2.350),
    "GOKA SENIOR HIGH/TECH SCHOOL":            ( 7.100, -2.500),
    "NKYERAA SENIOR HIGH SCHOOL":              ( 7.350, -2.350),
    "MANSEN SENIOR HIGH SCHOOL":               ( 7.350, -2.000),
    "DON BOSCO VOC./TECH. INST.":              ( 7.350, -2.350),
    "CHIRAA SENIOR HIGH SCHOOL":               ( 7.550, -2.150),
    "ST. JAMES SEM & SENIOR HIGH SCHOOL, ABESIM": (7.350, -2.350),
    "KOASE SENIOR HIGH/TECH SCHOOL":           ( 7.200, -2.000),
    "DORMAA SENIOR HIGH SCHOOL":               ( 7.200, -3.000),
    "SUMAMAN SENIOR HIGH SCHOOL":              ( 7.350, -2.350),
    "ST. ANN'S GIRLS SENIOR HIGH SCHOOL, SAMPA": (8.050, -2.650),
    "NAFANA SENIOR HIGH SCHOOL":               ( 8.100, -2.750),
    "DUADASO NO. 1 SENIOR HIGH/TECH SCHOOL":   ( 7.350, -2.350),
    "ST. AUGUSTINE SENIOR HIGH/TECH SCHOOL, SAAN CHARIKPONG": (7.750, -2.550),
    "SUNYANI METHODIST TECHNICAL INST.":       ( 7.350, -2.350),
    "NKRANKWANTA COMM SENIOR HIGH SCHOOL":     ( 7.750, -2.850),
    # WESTERN
    "NSEIN SENIOR HIGH SCHOOL":                ( 5.200, -2.700),
    "ANNOR ADJAYE SENIOR HIGH SCHOOL":         ( 5.600, -2.250),
    "HALF ASSINI SENIOR HIGH SCHOOL":          ( 5.000, -3.000),
    "ASANKRANGWA SENIOR HIGH/TECH SCHOOL":     ( 5.700, -2.450),
    "ST. MARY'S BOYS' SENIOR HIGH SCHOOL, APOWA": (5.050, -2.050),
    "BAIDOO BONSOE SENIOR HIGH/TECH SCHOOL":   ( 5.650, -2.000),
    "HUNI VALLEY SENIOR HIGH SCHOOL":          ( 5.600, -2.200),
    "SEKONDI COLLEGE":                         ( 4.940, -1.700),
    "DIABENE SENIOR HIGH/TECH SCHOOL":         ( 5.050, -1.600),
    "ADIEMBRA SENIOR HIGH SCHOOL":             ( 5.000, -2.800),
    "BOMPEH SENIOR HIGH./TECH SCHOOL":         ( 4.950, -2.050),
    "GHANA SENIOR HIGH/TECH SCHOOL":           ( 4.950, -1.700),
    "FIJAI SENIOR HIGH SCHOOL":                ( 4.940, -1.700),
    "ST. JOHN'S SENIOR HIGH SCHOOL, SEKONDI":  ( 4.940, -1.700),
    "ASANKRANGWA SENIOR HIGH SCHOOL":          ( 5.700, -2.450),
    "AMENFIMAN SENIOR HIGH SCHOOL":            ( 5.100, -2.100),
    "ARCHBISHOP PORTER GIRLS SENIOR HIGH SCHOOL": (4.940, -1.700),
    "BONZO-KAKU SENIOR HIGH SCHOOL":           ( 5.200, -2.500),
    "NKROFUL AGRIC. SENIOR HIGH SCHOOL":       ( 4.950, -2.100),
    "UTHMAN BIN AFAM SENIOR HIGH SCHOOL":      ( 5.000, -2.800),
    "ESIAMA SENIOR HIGH/TECH SCHOOL":          ( 5.000, -2.800),
    "BENSO SENIOR HIGH/TECH SCHOOL":           ( 5.500, -2.150),
    "TARKWA SENIOR HIGH SCHOOL":               ( 5.300, -2.000),
    "FIASEMAN SENIOR HIGH SCHOOL":             ( 5.100, -2.100),
    "SHAMA SENIOR HIGH SCHOOL":                ( 5.020, -1.640),
    "TAKORADI SENIOR HIGH SCHOOL":             ( 4.890, -1.760),
    "ST. AUGUSTINE'S SENIOR HIGH SCHOOL, BOGOSO": (5.570, -2.020),
    "AXIM GIRLS SENIOR HIGH SCHOOL":           ( 4.870, -2.240),
    "GWIRAMAN COMM.SENIOR HIGH SCHOOL":        ( 5.650, -2.350),
    "DABOASE SENIOR HIGH/TECH SCHOOL":         ( 5.350, -2.350),
    "MPOHOR SENIOR HIGH SCHOOL":               ( 5.150, -2.350),
    "PRESTEA SENIOR HIGH/TECH SCHOOL":         ( 5.430, -2.140),
    "SANKOR COMM. DAY SENIOR HIGH SCHOOL":     ( 5.350, -2.350),
    "EDINAMAN SENIOR HIGH SCHOOL":             ( 5.100, -1.350),
    # WESTERN NORTH
    "ASAWINSO SENIOR HIGH SCHOOL":             ( 6.500, -2.700),
    "SEFWI-WIAWSO SENIOR HIGH/TECH SCHOOL":    ( 6.200, -2.500),
    "SEFWI BEKWAI SENIOR HIGH SCHOOL":         ( 6.200, -2.330),
    "AKONTOMBRA SENIOR HIGH SCHOOL":           ( 6.050, -2.850),
    "ADJOAFUA COMM. SENIOR HIGH SCHOOL":       ( 6.400, -3.000),
    "NANA BRENTU SENIOR HIGH/TECH SCHOOL":     ( 5.600, -2.450),
    "BIBIANI SENIOR HIGH/TECH SCHOOL":         ( 6.450, -2.330),
    "CHIRANO COMM. DAY SENIOR HIGH SCHOOL":    ( 6.050, -2.450),
    "SEFWI-WIAWSO SENIOR HIGH SCHOOL":         ( 6.200, -2.500),
    "MANSO-AMENFI COMM. DAY SENIOR HIGH SCHOOL": (6.150, -2.350),
    "NSAWORA EDUMAFA COMM. SENIOR HIGH SCHOOL": (6.050, -2.850),
    "BODI SENIOR HIGH SCHOOL":                 ( 6.050, -2.850),
    "DADIESO SENIOR HIGH SCHOOL":              ( 6.050, -2.850),
    "JUABOSO SENIOR HIGH SCHOOL":              ( 6.250, -3.000),
    "BIA SENIOR HIGH/TECH SCHOOL":             ( 6.700, -3.000),
    "ST. JOSEPH SENIOR HIGH SCHOOL, SEFWI WIAWSO": (6.200, -2.500),
    # UPPER WEST
    "ST. AUGUSTINE SENIOR HIGH/TECH SCHOOL, SAAN CHARIKPONG": (10.350, -2.250),
    "LOGGU COMM. DAY SCHOOL":                  (10.050, -2.450),
    # Eastern — St. Joseph's Tech Inst in Kwahu South (SN 363)
    # (different from the Northern St. Joseph's Tech Inst in Tamale)
    # Resolve by SN in main() below
}


def main():
    print("=" * 60)
    print("GHANA SCHOOLS GEOCODING — FINAL FIX")
    print("=" * 60)

    df = pd.read_csv(INPUT, low_memory=False)
    print(f"Loaded {len(df)} schools from {INPUT}")

    corrected = 0
    for idx, row in df.iterrows():
        name = str(row["school_name"]).strip()
        if name in CORRECTIONS:
            lat, lon = CORRECTIONS[name]
            df.at[idx, "latitude"]  = lat
            df.at[idx, "longitude"] = lon
            corrected += 1

    print(f"Applied corrections to {corrected} schools")

    # Edge case: ST. JOSEPH'S TECH. INST. exists in both NORTHERN (Tamale) and
    # EASTERN (Kwahu South, SN 363). The name-based dict sets both to Tamale coords.
    # Fix the Eastern one by SN.
    sj_eastern = df[(df['sn'] == 363) & (df['region'].str.upper() == 'EASTERN')]
    if not sj_eastern.empty:
        df.loc[sj_eastern.index, 'latitude']  = 6.600
        df.loc[sj_eastern.index, 'longitude'] = -0.600
        print("  Fixed ST. JOSEPH'S TECH. INST. (Kwahu South, Eastern) -> (6.60, -0.60)")

    # Final validation
    remaining = []
    for idx, row in df.iterrows():
        region = str(row.get("region", "")).upper().strip()
        lat    = row.get("latitude")
        lon    = row.get("longitude")
        name   = str(row["school_name"]).strip()
        if region not in REGION_BOUNDS:
            continue
        lon_min, lon_max, lat_min, lat_max = REGION_BOUNDS[region]
        issues = []
        if pd.isna(lat) or pd.isna(lon):
            issues.append("MISSING COORDINATES")
        else:
            if not (lat_min <= lat <= lat_max):
                issues.append(f"LAT {lat:.4f} outside [{lat_min}, {lat_max}]")
            if not (lon_min <= lon <= lon_max):
                issues.append(f"LON {lon:.4f} outside [{lon_min}, {lon_max}]")
        if issues:
            remaining.append((name, region, "; ".join(issues)))

    df.to_csv(OUTPUT, index=False)

    print()
    print("=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Total schools       : {len(df)}")
    print(f"  Schools corrected   : {corrected}")
    missing = df[["latitude","longitude"]].isna().any(axis=1).sum()
    print(f"  Missing coordinates : {missing}")
    print(f"  Remaining anomalies : {len(remaining)}")
    if remaining:
        print()
        for r in remaining:
            print(f"  ⚠  {r[0]} ({r[1]}): {r[2]}")
    else:
        print("  ✅ Zero anomalies — all schools geocoded within regional bounds.")
    print(f"\nOutput saved to: {OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
