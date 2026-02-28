"""
Mission configurations for MarsAI multi-mission support.
Each mission has its own channel parameters, file types, and scientific references.
"""

MISSION_CONFIGS = {
    "mars": {
        "name": "Mars Rover — Perseverance",
        "delay_range": (3 * 60, 22 * 60),        # seconds
        "bandwidth_range": (0.1, 6.0),             # Mbps
        "default_delay": 13 * 60,                  # seconds
        "default_bandwidth": 2.0,
        "contact_window": None,                    # continuous
        "file_types": ["IMG", "CHEM", "ATM", "SEISM", "PIXL"],
        "high_value_refs": [
            "methane spike detected possible biosignature life on Mars",
            "organic compound carbon molecule biological origin",
            "water ice liquid brine ancient ocean lake river",
            "seismic activity marsquake tectonic interior structure",
            "unusual mineral deposit hematite sulfate phosphate",
            "chemical anomaly unexpected composition spike",
            "atmospheric pressure fluctuation gas release",
            "biosignature habitability life detection",
            "sedimentary rock water deposited ancient environment",
            "meteorite impact crater high energy event",
            "perchlorate concentration spike habitability",
            "manganese oxide biosignature relevant compound",
        ],
        "low_value_refs": [
            "routine standard survey no anomalies detected",
            "normal basalt volcanic rock common composition",
            "standard background measurement nominal conditions",
            "dust soil regolith common surface material",
            "routine terrain survey flat surface no features",
        ],
        "sensor_normal": {
            "temperature_mean": -25, "temperature_std": 8,
            "pressure_mean": 729, "pressure_std": 5,
            "chemical_index_max": 0.35,
        },
    },

    "satellite": {
        "name": "Earth Observation Satellite — LEO",
        "delay_range": (0.02, 0.05),               # seconds (20-50ms)
        "bandwidth_range": (10.0, 150.0),           # Mbps
        "default_delay": 0.03,
        "default_bandwidth": 45.0,
        "contact_window": 600,                     # 10 min window per orbit
        "file_types": ["SAR", "OPT", "THERMAL", "RADAR", "MULTISPECT"],
        "high_value_refs": [
            "flood disaster emergency mapping river overflow",
            "wildfire ignition thermal anomaly smoke detection",
            "illegal deforestation land use change forest loss",
            "earthquake damage assessment urban destruction",
            "oil spill marine pollution coastal contamination",
            "glacier retreat ice sheet melting climate change",
            "crop failure drought stress vegetation index",
            "military vessel unusual maritime activity",
            "volcanic eruption ash cloud lava flow",
            "tsunami inundation coastal flooding",
        ],
        "low_value_refs": [
            "routine coverage pass nominal conditions",
            "standard agricultural monitoring seasonal",
            "cloud cover obscured no useful data",
            "ocean surface standard conditions no anomaly",
            "urban area routine monitoring no change",
        ],
        "sensor_normal": {
            "temperature_mean": 20, "temperature_std": 15,
            "pressure_mean": 1013, "pressure_std": 10,
            "chemical_index_max": 0.3,
        },
    },

    "lunar": {
        "name": "Lunar Mission — Artemis Base Camp",
        "delay_range": (1.2, 1.4),                 # seconds
        "bandwidth_range": (1.0, 20.0),             # Mbps
        "default_delay": 1.3,
        "default_bandwidth": 8.0,
        "contact_window": None,
        "file_types": ["IMG", "DRILL", "SEISM", "RADAR", "THERMAL"],
        "high_value_refs": [
            "water ice confirmed depth ISRU resource utilization",
            "moonquake seismic activity structural interior",
            "lava tube void subsurface habitat potential",
            "regolith composition oxygen extraction ISRU",
            "solar wind implanted helium-3 fusion resource",
            "impact melt ancient volcanic activity",
            "permanently shadowed region ice deposits",
            "rare earth elements mineral concentration",
        ],
        "low_value_refs": [
            "routine surface imaging standard regolith",
            "nominal temperature diurnal variation expected",
            "standard background radiation measurement",
            "routine drill sample common anorthosite",
        ],
        "sensor_normal": {
            "temperature_mean": -20, "temperature_std": 50,
            "pressure_mean": 0, "pressure_std": 0,
            "chemical_index_max": 0.3,
        },
    },

    "deepspace": {
        "name": "Deep Space Probe — Voyager Class",
        "delay_range": (3600, 28800),              # seconds (1-8 hours)
        "bandwidth_range": (0.001, 0.08),           # Mbps (very slow!)
        "default_delay": 14400,                    # 4 hours
        "default_bandwidth": 0.04,
        "contact_window": 8 * 3600,               # 8h DSN window per day
        "file_types": ["PLASMA", "MAG", "COSMIC", "RADIO", "PARTICLE"],
        "high_value_refs": [
            "heliosphere boundary crossing termination shock",
            "interstellar medium plasma density anomaly",
            "high energy cosmic ray burst galactic origin",
            "magnetic field reversal unexpected polarity",
            "solar wind interaction bow shock detection",
            "interstellar neutral particle detection",
            "anomalous cosmic ray acceleration",
            "hydrogen wall heliosheath boundary",
        ],
        "low_value_refs": [
            "nominal plasma density standard solar wind",
            "background cosmic ray flux standard level",
            "routine magnetic field measurement nominal",
            "standard particle flux background reading",
        ],
        "sensor_normal": {
            "temperature_mean": -270, "temperature_std": 1,
            "pressure_mean": 0, "pressure_std": 0,
            "chemical_index_max": 0.25,
        },
    },
}

# File templates per mission
MISSION_FILE_TEMPLATES = {
    "mars": [
        {
            "type": "IMG", "name_prefix": "ZCAM_SOL",
            "descriptions": [
                "Unusual reddish-brown mineral deposit detected near rock formation Delta-7, possible hematite concretions",
                "Routine terrain survey image, flat dust-covered basalt surface, no notable features",
                "Layered sedimentary rock outcrop with visible stratification — potential water-deposited origin",
                "Dust devil track visible in surface regolith, 340m in length heading northwest",
                "Close-up of rock abrasion target showing fine crystalline structure with sulfate veins",
                "Panoramic horizon scan, no anomalies detected, standard sol documentation",
            ],
            "size_range": (8, 45), "ext": "jpg",
            "chemical_index_base": 0.08,
        },
        {
            "type": "CHEM", "name_prefix": "CHEMCAM_RDRS",
            "descriptions": [
                "LIBS spectroscopy result: elevated manganese oxide content — biosignature-relevant compound",
                "Standard basalt composition: SiO2 48%, FeO 18%, MgO 9% — no anomalies",
                "Perchlorate concentration spike detected: 0.6% by weight, significant for habitability research",
                "Calcium sulfate vein analysis: gypsum variant, consistent with ancient aqueous environment",
                "Routine rock classification complete: olivine-rich basalt, volcanic origin confirmed",
                "Organic carbon detection attempt — inconclusive signal, requires follow-up with SAM",
            ],
            "size_range": (2, 12), "ext": "csv",
            "chemical_index_base": 0.18,
        },
        {
            "type": "ATM", "name_prefix": "MEDA_ATMO",
            "descriptions": [
                "Sudden methane spike: 21 ppb above baseline at 18:32 local time — high scientific priority",
                "Standard atmospheric pressure reading: 730 Pa, temperature -45°C, wind 4.2 m/s NNE",
                "Dust storm onset detected — opacity index rising from 0.4 to 1.2 over 3 hours",
                "UV radiation flux measurement: solar irradiance 586 W/m², standard sol conditions",
                "Unusual pressure oscillation pattern detected — possible sub-surface gas release",
                "Routine morning atmospheric profile: inversion layer at 800m altitude",
            ],
            "size_range": (1, 8), "ext": "dat",
            "chemical_index_base": 0.12,
        },
        {
            "type": "SEISM", "name_prefix": "SEIS_EVENT",
            "descriptions": [
                "Marsquake magnitude 3.2 detected at 11:14 — strongest seismic event this sol, epicenter 847km",
                "Low-frequency seismic noise: likely thermal contraction of surface regolith at sunset",
                "High-frequency seismic signal detected — possible meteorite impact 200km north",
                "Background seismic monitoring: nominal levels, no events above M1.0 this period",
            ],
            "size_range": (3, 20), "ext": "dat",
            "chemical_index_base": 0.10,
        },
        {
            "type": "PIXL", "name_prefix": "PIXL_XRF",
            "descriptions": [
                "Potential biosignature indicator: organic sulfur compound cluster detected in rock matrix",
                "X-ray fluorescence: iron-magnesium silicate, standard mafic composition, low priority",
                "Phosphate mineral identification: apatite group — relevant to prebiotic chemistry research",
                "Carbonate mineral vein: siderite composition, formed in ancient CO2-rich water",
            ],
            "size_range": (5, 25), "ext": "csv",
            "chemical_index_base": 0.20,
        },
    ],

    "satellite": [
        {
            "type": "SAR", "name_prefix": "SAR_PASS",
            "descriptions": [
                "Flash flood detected — river delta overflow 340km², emergency response mapping required",
                "Routine SAR coverage pass, agricultural fields nominal, no surface changes detected",
                "Urban subsidence detected: 3.2cm displacement in residential district — structural risk",
                "Standard ocean surface backscatter, wind speed 12 knots, no anomalies",
            ],
            "size_range": (40, 200), "ext": "tif",
            "chemical_index_base": 0.10,
        },
        {
            "type": "OPT", "name_prefix": "OPT_IMG",
            "descriptions": [
                "Wildfire ignition point confirmed — thermal anomaly 47°C above background, smoke plume NE",
                "Routine optical pass — 60% cloud cover, limited surface visibility, low data value",
                "Deforestation event detected: 12km² primary forest loss since last pass",
                "Standard crop monitoring: NDVI nominal, seasonal growth expected",
            ],
            "size_range": (80, 400), "ext": "tif",
            "chemical_index_base": 0.08,
        },
        {
            "type": "THERMAL", "name_prefix": "THERMAL_IR",
            "descriptions": [
                "Oil spill thermal signature confirmed — 8km² slick detected off coastal zone",
                "Volcanic heat flux anomaly — 340°C hotspot detected on caldera rim",
                "Standard urban heat island mapping, nominal diurnal pattern",
                "Routine sea surface temperature pass, no anomalies detected",
            ],
            "size_range": (30, 150), "ext": "dat",
            "chemical_index_base": 0.15,
        },
        {
            "type": "MULTISPECT", "name_prefix": "MS_BAND",
            "descriptions": [
                "Crop stress detected — NDWI indicates severe drought affecting 200km² farmland",
                "Standard multispectral vegetation index, seasonal baseline, no anomalies",
                "Glacier retreat confirmed: 2.3km recession since reference image",
                "Routine coastal monitoring, water quality nominal",
            ],
            "size_range": (100, 500), "ext": "tif",
            "chemical_index_base": 0.12,
        },
    ],

    "lunar": [
        {
            "type": "IMG", "name_prefix": "LCAM_SOL",
            "descriptions": [
                "Lava tube entrance detected — subsurface void 40m diameter, potential habitat site",
                "Routine surface imaging, standard regolith, no notable geological features",
                "Permanently shadowed region boundary mapped — ice deposit indicators present",
                "Standard crater morphology survey, nominal ejecta distribution",
            ],
            "size_range": (5, 30), "ext": "jpg",
            "chemical_index_base": 0.08,
        },
        {
            "type": "DRILL", "name_prefix": "DRILL_CORE",
            "descriptions": [
                "Water ice confirmed at 2.3m depth — 8% concentration by mass, significant ISRU discovery",
                "Standard anorthosite composition at 1.5m depth — common highland material",
                "Helium-3 implanted concentration: 12ppb — potential fusion fuel resource",
                "Routine regolith sample: standard silicate composition, no volatile deposits",
            ],
            "size_range": (2, 15), "ext": "csv",
            "chemical_index_base": 0.15,
        },
        {
            "type": "SEISM", "name_prefix": "LSEIS_EVT",
            "descriptions": [
                "Shallow moonquake M2.8 detected — possible active fault system, structural survey needed",
                "Deep moonquake 900km depth — standard tidal origin, background level",
                "Meteorite impact seismic signature — M1.5 equivalent, 12km from base",
                "Thermal contraction creak: daily surface cooling signal, nominal",
            ],
            "size_range": (3, 18), "ext": "dat",
            "chemical_index_base": 0.10,
        },
    ],

    "deepspace": [
        {
            "type": "PLASMA", "name_prefix": "PLASMA_MEAS",
            "descriptions": [
                "Unexpected plasma density spike 340% above baseline — possible heliosphere boundary crossing",
                "Nominal solar wind plasma density: 5 particles/cm³, standard interplanetary conditions",
                "Termination shock crossing confirmed — velocity drop from 400 to 100 km/s detected",
                "Routine plasma frequency measurement, background interstellar medium level",
            ],
            "size_range": (0.1, 2), "ext": "dat",
            "chemical_index_base": 0.12,
        },
        {
            "type": "MAG", "name_prefix": "MAG_FIELD",
            "descriptions": [
                "Magnetic field polarity reversal detected — unexpected interstellar boundary signature",
                "Standard interplanetary magnetic field: 0.1nT, nominal Parker spiral configuration",
                "Magnetic reconnection event detected — energy release 10^18 joules",
                "Routine background measurement, nominal heliospheric current sheet",
            ],
            "size_range": (0.05, 1), "ext": "dat",
            "chemical_index_base": 0.08,
        },
        {
            "type": "COSMIC", "name_prefix": "CRS_DETECT",
            "descriptions": [
                "High-energy cosmic ray burst: 10^20 eV — galactic origin confirmed, rare ultra-high-energy event",
                "Standard galactic cosmic ray flux: background level, no anomalies",
                "Anomalous cosmic ray acceleration detected — possible nearby pulsar wind nebula",
                "Routine particle flux measurement, solar modulation nominal",
            ],
            "size_range": (0.2, 3), "ext": "dat",
            "chemical_index_base": 0.15,
        },
    ],
}
