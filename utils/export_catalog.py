"""
Catalog and Statistics Exporter
===============================
Extracts metadata from all 4 datasets (nsrdb, edb, vfdb, sddb)
and outputs structured CSVs and JSON into the metadata/ folder.
"""

import os
import csv
import json
import re

DATASETS = [
    {
        'id': 'nsrdb',
        'name': 'MIT-BIH Normal Sinus Rhythm Database',
        'cat': 'Control (Ritmo Sinusal Normal)',
        'path': 'data/nsrdb',
        'desc': 'Sujetos sanos sin arritmias significativas'
    },
    {
        'id': 'edb',
        'name': 'European ST-T Database',
        'cat': 'Isquemia Miocárdica (Cambios ST-T)',
        'path': 'data/edb',
        'desc': 'Pacientes con angina, infarto previo y enf. coronaria (cambios ST y onda T)'
    },
    {
        'id': 'vfdb',
        'name': 'MIT-BIH Malignant Ventricular Ectopy Database',
        'cat': 'Arritmias Ventriculares Letales / Malignas',
        'path': 'data/vfdb',
        'desc': 'Episodios de taquicardia ventricular sostenida, flutter y fibrilación ventricular'
    },
    {
        'id': 'sddb',
        'name': 'Sudden Cardiac Death Holter Database',
        'cat': 'Paro Cardíaco / Muerte Súbita Cardíaca (MSC)',
        'path': 'data/sddb',
        'desc': 'Registros Holter completos hasta el momento del colapso / parada cardíaca'
    }
]


def generate_catalogs():
    os.makedirs('metadata', exist_ok=True)
    all_records = []

    for db in DATASETS:
        rec_file = os.path.join(db['path'], 'RECORDS')
        records = []
        if os.path.exists(rec_file):
            with open(rec_file, 'r', encoding='utf-8') as f:
                records = [l.strip() for l in f if l.strip()]

        for r in records:
            hea_file = os.path.join(db['path'], f"{r}.hea")
            dat_file = os.path.join(db['path'], f"{r}.dat")
            atr_file = os.path.join(db['path'], f"{r}.atr")
            ari_file = os.path.join(db['path'], f"{r}.ari")
            xws_file = os.path.join(db['path'], f"{r}.xws")
            qrs_file = os.path.join(db['path'], f"{r}.qrs")

            dat_size = os.path.getsize(dat_file) if os.path.exists(dat_file) else 0
            has_atr = os.path.exists(atr_file)
            has_ari = os.path.exists(ari_file)
            has_xws = os.path.exists(xws_file)
            has_qrs = os.path.exists(qrs_file)

            fs = 0.0
            n_sig = 0
            n_samples = 0
            start_time = ""
            channels = []
            gains = []
            comments = []
            age = ""
            sex = ""

            if os.path.exists(hea_file):
                with open(hea_file, 'r', errors='ignore') as f:
                    lines = [l.strip() for l in f if l.strip()]
                if lines:
                    first = lines[0].split()
                    n_sig = int(first[1]) if len(first) > 1 else 0
                    fs = float(first[2]) if len(first) > 2 else 0.0
                    n_samples = int(first[3]) if len(first) > 3 else 0
                    if len(first) > 4:
                        start_time = first[4]

                    for l in lines[1:1 + n_sig]:
                        parts = l.split()
                        lead_name = parts[8] if len(parts) >= 9 else parts[-1]
                        channels.append(lead_name)
                        if len(parts) >= 3:
                            gains.append(parts[2])

                    for l in lines:
                        if l.startswith('#'):
                            comment_text = l.lstrip('#').strip()
                            if comment_text:
                                comments.append(comment_text)

            for c in comments:
                m_age = re.search(r'Age:\s*(\d+)', c, re.IGNORECASE)
                if m_age:
                    age = m_age.group(1)
                m_sex = re.search(r'Sex:\s*([MF])', c, re.IGNORECASE)
                if m_sex:
                    sex = m_sex.group(1)
                m_nsr = re.match(r'^(\d+)\s+([MF])$', c)
                if m_nsr:
                    age = m_nsr.group(1)
                    sex = m_nsr.group(2)

            dur_sec = n_samples / fs if fs > 0 else 0
            dur_h = dur_sec / 3600.0
            dur_str = f"{int(dur_sec // 3600):02d}:{int((dur_sec % 3600) // 60):02d}:{int(dur_sec % 60):02d}"

            row = {
                'database': db['id'],
                'database_name': db['name'],
                'clinical_category': db['cat'],
                'record_id': r,
                'sampling_rate_hz': fs,
                'num_channels': n_sig,
                'channels': '/'.join(channels),
                'gain_adc_per_mv': '/'.join(gains),
                'num_samples': n_samples,
                'start_time': start_time,
                'duration_seconds': round(dur_sec, 2),
                'duration_hours': round(dur_h, 2),
                'duration_formatted': dur_str,
                'patient_age': age,
                'patient_sex': sex,
                'clinical_notes': ' ; '.join(comments),
                'has_atr': has_atr,
                'has_ari': has_ari,
                'has_xws': has_xws,
                'has_qrs': has_qrs,
                'dat_size_mb': round(dat_size / (1024 * 1024), 2)
            }
            all_records.append(row)

    # Write master catalog
    fields = list(all_records[0].keys())
    with open('metadata/master_records_catalog.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)

    # Write per-database catalogs
    for db in DATASETS:
        db_rows = [r for r in all_records if r['database'] == db['id']]
        with open(f"metadata/{db['id']}_catalog.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(db_rows)

    # Write summary statistics JSON
    summary = {}
    for db in DATASETS:
        db_rows = [r for r in all_records if r['database'] == db['id']]
        total_samples = sum(r['num_samples'] for r in db_rows)
        total_hours = sum(r['duration_hours'] for r in db_rows)
        total_mb = sum(r['dat_size_mb'] for r in db_rows)
        fs_list = sorted(list(set(r['sampling_rate_hz'] for r in db_rows)))
        summary[db['id']] = {
            'name': db['name'],
            'category': db['cat'],
            'description': db['desc'],
            'num_records': len(db_rows),
            'sampling_rate_hz': fs_list,
            'total_duration_hours': round(total_hours, 2),
            'total_samples': total_samples,
            'total_raw_data_mb': round(total_mb, 2),
            'records_list': [r['record_id'] for r in db_rows]
        }

    summary['global'] = {
        'total_databases': len(DATASETS),
        'total_records': len(all_records),
        'total_duration_hours': round(sum(r['duration_hours'] for r in all_records), 2),
        'total_raw_data_mb': round(sum(r['dat_size_mb'] for r in all_records), 2)
    }

    with open('metadata/summary_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(all_records)} records to metadata/master_records_catalog.csv and metadata/summary_statistics.json")


if __name__ == '__main__':
    generate_catalogs()
