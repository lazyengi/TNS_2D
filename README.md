# TNS-2D

Simulatore 2D di una rete di sensori che osserva **oggetti estesi** in movimento. Ogni oggetto ha una sagoma planare (libreria di 21 velivoli), una traiettoria a uniciclo e produce, a ogni scansione, una nuvola di punti etichettata: rilevazioni vere più clutter.

Il progetto fa tre cose:

1. **Simulare uno scenario** (più oggetti, più sensori) e salvarne un video MP4.
2. **Generare un dataset** di nuvole di punti etichettate con l’indice di sagoma, per un compito di classificazione.
3. **Costruire la libreria di sagome** a partire da un file HDF5 di vertici, con contorno, superficie triangolata e tabella di visibilità (linea di vista).

Non è un pacchetto installabile. Gli script aggiungono la radice del repository a `sys.path` e importano `src.tns_2d`.

## Requisiti

- Python 3.12
- I pacchetti in [`requirements.txt`](requirements.txt): NumPy, pandas, Matplotlib, tqdm, h5py, `triangle` (triangolazione di Shewchuk)
- [FFmpeg](https://ffmpeg.org/) sul `PATH`, solo per il video. Il renderer usa `FFMpegWriter` con codec `libx264`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verifica FFmpeg con `ffmpeg -version`.

## Layout

```
TNS_2D/
├── scripts/run_simulation.py          # scenario di esempio + video
├── utils/generate_dataset.py          # dataset di classificazione
├── utils/generate_shape_library.py    # da HDF5 a pickle della libreria
├── test.py                            # controllo visivo della triangolazione
├── src/tns_2d/
│   ├── simulation/
│   │   ├── data_structures.py         # ExtendedObject, Sensor
│   │   └── simulator.py               # dinamica, clutter, rilevazioni
│   └── graphics/
│       └── rendering_engine.py        # animazione Matplotlib → MP4
├── data/
│   ├── raw_aircraft_shape_library.h5  # vertici grezzi
│   ├── aircraft_shape_library.pkl     # libreria usata dalla simulazione
│   └── shape_library.pkl              # stessa libreria, usata dal dataset
└── output/
    ├── simulation_output.mp4
    └── full_dataset.pkl
```

Il modello di calcolo, i frame di riferimento e il formato dei file sono in [`docs/architecture.md`](docs/architecture.md).

## Simulazione e video

Lo scenario predefinito è in `scripts/run_simulation.py`: quadrato di lato 100 m (`SPINE_LENGTH = 50`), tre velivoli, tre sensori in modalità `center`, durata massima 60 s, passo `1/120` s.

| Oggetto | `shape_idx` | Sagoma | Ingresso |
|---|---|---|---|
| 0 | 2 | F-117 | da destra, a t = 2 s |
| 1 | 1 | F-35 | da sinistra, a t = 4 s |
| 2 | 4 | JAS 39 | dal basso, a t = 5 s |

I tre sensori stanno su una circonferenza di raggio `0.98 * SPINE_LENGTH`, hanno campo di vista ±35°, portata 5–50 m, `pd = 0.75`, dropout di Markov (`alpha = 0.8`) e clutter `0.05` punti al metro quadro.

```powershell
python scripts/run_simulation.py
```

Il video finisce in `output/simulation_output.mp4`. Il rendering a 60 fps e 120 dpi con preset `veryslow` è lento: è una codifica lossless (`-crf 0`), non un’anteprima.

Per cambiare lo scenario si editano, nello stesso script, i dizionari `extended_object_definitions`, `sensor_definitions`, `simulation_config` e `rendering_config`. I nomi dei campi coincidono con i costruttori di `ExtendedObject`, `Sensor` e `Simulator`.

### Campi di un oggetto

| Campo | Unità | Significato |
|---|---|---|
| `identifier` | — | Etichetta scritta sulle rilevazioni vere |
| `shape_idx` | — | Indice nella libreria di sagome |
| `t_init` | s | Istante di comparsa sulla griglia globale |
| `initial_position` | m | `[x, y]` nel frame mondo |
| `initial_heading` | rad | Prua; 0 è l’asse +x |
| `initial_speed` | m/s | Velocità iniziale |
| `long_vals`, `long_times` | m/s², s | Accelerazione longitudinale, costante a tratti |
| `centr_vals`, `centr_times` | m/s², s | Accelerazione centripeta, costante a tratti |
| `dt` | s | Passo della traiettoria grezza |

`long_times` e `centr_times` sono sul **tempo proprio dell’oggetto**, che parte da 0 alla comparsa. Non sono tempi assoluti di simulazione. Con un solo campione il comando resta costante per tutta la traiettoria. Nello script di esempio `g2mss` converte i valori da *g* e `mach2mss` converte la velocità da Mach (343 m/s).

### Campi di un sensore

| Campo | Unità | Significato |
|---|---|---|
| `generation_mode` | — | `"contour"`, `"surface"` o `"center"` |
| `position`, `heading` | m, rad | Posa fissa; la prua è l’asse del campo di vista |
| `min_range`, `max_range` | m | Portata |
| `min_bearing`, `max_bearing` | rad | Bearing rispetto alla prua del sensore |
| `pd` | — | Probabilità che la scansione sia attiva |
| `alpha` | — | Persistenza del dropout di Markov; `None` seleziona il modello di Bernoulli |
| `clutter_rate` | 1/m² | Densità media del clutter |
| `sigma_v` | m | Deviazione standard del rumore di misura, isotropo |
| `resolution` | vedi sotto | Intensità del processo di Poisson sulle rilevazioni |
| `dt_sample` | s | Periodo di scansione |
| `t_offset` | s | Ritardo della prima scansione |

`resolution` cambia significato con la modalità: rilevazioni attese **per metro di spigolo** in `contour`, **per metro quadro di triangolo** in `surface`. In `center` non è usata.

`dt_sample` e `t_offset` vanno scelti come multipli interi di `global_dt`: gli indici di campionamento sono `round(t / global_dt)`.

### Flag del simulatore

| Flag | Effetto se vero |
|---|---|
| `use_dropouts` | Applica Bernoulli o Markov. Se falso, ogni scansione è attiva |
| `use_los` | Applica l’occlusione propria. Ha effetto solo in modalità `contour` |
| `use_oor` | Scarta la geometria fuori dal campo di vista |
| `verbose` | Stampa i tre passi della pipeline |

`spine_length` è la semi-lunghezza del quadrato di volo: un punto con `|x|` o `|y|` maggiori esce di scena. `max_duration` tronca la simulazione anche se gli oggetti sono ancora dentro.

### Configurazione del video

| Chiave | Ruolo |
|---|---|
| `extended_object_colors`, `sensor_colors` | Colori RGB in `[0, 1]` |
| `extended_object_censor_levels`, `sensor_censor_levels` | `"null"` (opaco), `"mild"` (alpha 0.3), `"total"` (nascosto) |
| `alpha` | Fattore extra sull’oggetto, moltiplicato dal livello di censura |
| `linewidth`, `min_linewidth` | `linewidth` è lo spessore della traccia alla creazione. `min_linewidth` arriva alla manutenzione dei segmenti e non cambia il disegno |
| `clutter_color_override` | Colore unico del clutter; `None` usa il colore del sensore |
| `MP4_PATH`, `DURATION_S`, `SPINE_LENGTH` | Destinazione, durata e scala degli assi |
| `FIGSIZE`, `DPI`, `FPS` | Figura Matplotlib e cadenza del filmato |

Nel filmato i quadratini sono rilevazioni vere, i punti piccoli sono clutter. Le nuvole svaniscono in 1,5 s. Il cuneo è il campo di vista.

## Dataset di classificazione

`utils/generate_dataset.py` produce esempi indipendenti: un solo oggetto fermo, un sensore nell’origine in modalità `contour`, una sola scansione. La classe è `shape_idx`.

Parametri in cima al file:

| Costante | Default nel codice | Effetto |
|---|---|---|
| `NUM_EXAMPLES` | `1_000_000` | Tentativi di generazione |
| `USE_EXTENDED_OBJECT_COORDINATES` | `True` | Nuvola nel frame dell’oggetto |
| `USE_RANDOM_DISTANCE` | `False` | `False`: distanza fissa `POSE_DISTANCE`. `True`: raggio uniforme in area nell’anello `[MIN_DISTANCE, 2 * POSE_DISTANCE]` |
| `POSE_DISTANCE`, `MIN_DISTANCE` | 50 m, 15 m | Geometria posa–sensore |
| `USE_LOS` | `True` | Auto-occlusione del contorno |
| `SENSOR_RESOLUTION`, `SENSOR_NOISE_STD` | 2 m⁻¹, 0.1 m | Densità e rumore |
| `VALIDATE_AFTER_GENERATION` | `True` | Apre una finestra Matplotlib a fine corsa |
| `NUM_VALIDATION_EXAMPLES` | 100 | Esempi mostrati; un tasto passa al successivo |

```powershell
python utils/generate_dataset.py
```

Con un milione di esempi il file è dell’ordine del gigabyte e la validazione è interattiva. Per una prova, abbassa `NUM_EXAMPLES` e metti `VALIDATE_AFTER_GENERATION = False`.

`USE_RANDOM_DISTANCE = True` con i default attuali mette l’oggetto fino a 100 m dall’origine, mentre il `Simulator` interno usa `spine_length = 60`. Oltre 60 m la traccia è azzerata e l’esempio viene scartato. Per usare l’anello intero, `spine_length` in `generate_single_example` deve coprire `2 * POSE_DISTANCE`.

Un esempio fallito (nessun punto misurato) viene scartato: `len(data)` può essere minore di `num_examples`.

### File già presente in `output/full_dataset.pkl`

Quel pickle **non** è stato scritto dalla versione attuale dello script. Contiene 100 000 esempi e usa i nomi della revisione precedente, quando l’oggetto si chiamava ancora *aircraft*:

```python
import pickle
with open("output/full_dataset.pkl", "rb") as f:
    full = pickle.load(f)

full["metadata"]["use_aircraft_coordinates"]  # True
full["data"][0].keys()  # point_cloud, shape_idx, sensor_pos_ac
```

| | Pickle in repository | Script attuale |
|---|---|---|
| Esempi | 100 000 | `NUM_EXAMPLES = 1_000_000` |
| Flag coordinate | `use_aircraft_coordinates` | `use_extended_object_coordinates` |
| Posa del sensore nel frame oggetto | `sensor_pos_ac` | `sensor_pos_eoc` |

Geometria e rumore coincidono con i default attuali: distanza fissa 50 m, risoluzione 2 m⁻¹, rumore 0.1 m, linea di vista attiva, coordinate dell’oggetto. `point_cloud` è un array `(N, 2)`.

Lo script attuale, con `USE_EXTENDED_OBJECT_COORDINATES = True`, salva:

```python
{
    "point_cloud": np.ndarray,   # (N, 2), frame dell'oggetto
    "shape_idx": int,
    "sensor_pos_eoc": np.ndarray # (2,), sensore visto dall'oggetto
}
```

Con il flag a `False` salva invece `extended_object_position`, `extended_object_heading` e `sensor_position` (il sensore resta nell’origine) e la nuvola resta nel frame mondo.

## Libreria di sagome

Entrambe le copie pickle sono una lista di 21 dizionari, uno per velivolo, nello stesso ordine. L’indice di lista è `shape_idx`.

| idx | id | nome |
|---|---|---|
| 0 | A380 | Airbus A380 |
| 1 | F35 | Lockheed Martin F-35 Lightning II |
| 2 | F117 | Lockheed F-117 Nighthawk |
| 3 | J20 | Chengdu J-20 Mighty Dragon |
| 4 | JAS39 | Saab JAS 39 Gripen |
| 5 | MIG25 | Mikoyan-Gurevich MiG-25 Foxbat |
| 6 | MIG27sw | MiG-27 Flogger D, ala a freccia |
| 7 | MIG27 | MiG-27 Flogger D |
| 8 | SPIT | Supermarine Spitfire |
| 9 | SU25 | Sukhoi Su-25 Frogfoot |
| 10 | SU47 | Sukhoi Su-47 Berkut |
| 11 | A10 | Fairchild Republic A-10 Thunderbolt II |
| 12 | X32 | Boeing X-32 |
| 13 | YF23 | Northrop YF-23 |
| 14 | AV8B | McDonnell Douglas AV-8B Harrier II |
| 15 | B2 | Northrop Grumman B-2 Spirit |
| 16 | 172 | Cessna 172 Skyhawk |
| 17 | DM2K | Dassault Mirage 2000 |
| 18 | EFT | Eurofighter Typhoon |
| 19 | F14sw | Grumman F-14 Tomcat, ala a freccia |
| 20 | F14 | Grumman F-14 Tomcat |

```python
import pickle
with open("data/shape_library.pkl", "rb") as f:
    library = pickle.load(f)

entry = library[2]
entry["shortName"]            # "F117"
entry["contourShapeMatrix"]   # (5, n_spigoli)
entry["surfaceShapeMatrix"]   # (7, n_triangoli)
entry["LoSTable"]["flags"]    # (3939, n_spigoli), uint8
```

`scripts/run_simulation.py` legge `data/aircraft_shape_library.pkl`. `utils/generate_dataset.py` e `test.py` leggono `data/shape_library.pkl`. I due pickle in repository hanno lo stesso contenuto; vanno tenuti allineati se si rigenera la libreria.

### Rigenerare la libreria

`utils/generate_shape_library.py` legge `data/raw_aircraft_shape_library.h5` e scrive `data/aircraft_shape_library.pkl`. Ogni gruppo HDF5 ha il dataset `foldedVertices` (metà sagoma, simmetrica rispetto all’asse x) e gli attributi `name` e `id`.

Nel file ci sono due funzioni con lo **stesso nome** `contourShapeMatrixGenerator`. La seconda, quella che costruisce la superficie, sostituisce la prima. La chiamata che dovrebbe costruire il contorno dai vertici piegati esegue quindi la triangolazione e il generatore, così com’è, non ricostruisce la libreria. Per ripristinarlo la seconda definizione va rinominata (il ruolo è quello di `surfaceShapeMatrixGenerator`) e la seconda chiamata nel `main` va aggiornata di conseguenza.

Anche a funzione sistemata, i flag in cima al file sono tutti `True`, incluso `FORCE_REGENERATION`. L’esecuzione ricalcola le tabelle di linea di vista (migliaia di punti per sagoma) e apre finestre Matplotlib che avanzano con un tasto. Per il solo ricalcolo senza plot: `RUN_TRIANGULATION_VALIDATION`, `RUN_GRID_VALIDATION`, `RUN_PERFORMANCE_VALIDATION` e `RUN_VISUAL_VALIDATION` a `False`.

`test.py` non è una suite automatica. Va lanciato dalla radice del repository, carica `data/shape_library.pkl`, triangola una sagoma a caso e la disegna.

## Uso come libreria

Dalla radice del repository:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -c "from src.tns_2d.simulation.simulator import Simulator"
```

Dopo `simulator.run()`:

- `simulator.time_grid` è la griglia temporale comune.
- `extended_object.aligned_track` ha forma `(4, K)`: righe `x`, `y`, prua, velocità. I campioni fuori dal quadrato sono `NaN`.
- `sensor.dataset` è un dizionario `times`, `measurements`, `labels`. Ogni misura è un array `(N, 2)` nel frame mondo. L’etichetta `-1` è clutter; un intero `≥ 0` è l’`identifier` dell’oggetto. Le scansioni senza punti non compaiono in `times`.
- `sensor.dropout_sequence` vale `1` se la scansione è attiva e `0` se è in dropout. La sua lunghezza è il numero di istanti di campionamento del sensore, non il numero di righe poi salvate nel dataset.

Il rendering si invoca con `render_animation(simulator, rendering_config)` da `src.tns_2d.graphics.rendering_engine`.
