"""
ECG Loader Utility for PhysioNet WFDB Datasets
==============================================
Provides pure-Python functionality to load, parse, and process ECG records
from PhysioNet (Format 212 binary signals, WFDB headers, and annotations).

Requires NO third-party packages (no pip, no wfdb required). Standard Python only.
Supports:
  - NSRDB (MIT-BIH Normal Sinus Rhythm Database)
  - EDB   (European ST-T Database)
  - VFDB  (MIT-BIH Malignant Ventricular Ectopy Database)
  - SDDB  (Sudden Cardiac Death Holter Database)
"""

import os
import re
import sys
import json
import csv
from typing import Dict, List, Tuple, Optional, Any


class ECGRecord:
    """Represents a loaded ECG record with signals, annotations, and metadata."""

    def __init__(self, record_path: str):
        self.record_path = os.path.splitext(record_path)[0]
        self.record_name = os.path.basename(self.record_path)
        self.header: Dict[str, Any] = {}
        self.signals: List[List[float]] = []  # [channel_index][sample_index]
        self.raw_signals: List[List[int]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.load_header()

    def load_header(self) -> Dict[str, Any]:
        """Parses the WFDB .hea file."""
        hea_file = f"{self.record_path}.hea"
        if not os.path.exists(hea_file):
            raise FileNotFoundError(f"Header file not found: {hea_file}")

        with open(hea_file, 'r', errors='ignore') as f:
            lines = [l.strip() for l in f if l.strip()]

        first = lines[0].split()
        n_sig = int(first[1]) if len(first) > 1 else 0
        fs = float(first[2]) if len(first) > 2 else 0.0
        n_samples = int(first[3]) if len(first) > 3 else 0
        start_time = first[4] if len(first) > 4 else ""

        channels = []
        comments = []

        for line in lines[1:1 + n_sig]:
            parts = line.split()
            # Standard line: filename format gain[(baseline)]/units ... lead
            fmt = parts[1] if len(parts) > 1 else "212"
            gain_field = parts[2] if len(parts) > 2 else "200"
            lead_name = parts[8] if len(parts) >= 9 else parts[-1]

            gain = 200.0
            baseline = 0.0
            units = "mV"

            # Parse gain e.g. "200", "200(0)/mV", "800/mV", "0"
            m = re.match(r'^([0-9.]+)(?:\(([0-9.-]+)\))?(?:/(.+))?$', gain_field)
            if m:
                g_val = float(m.group(1))
                if g_val > 0:
                    gain = g_val
                if m.group(2):
                    baseline = float(m.group(2))
                if m.group(3):
                    units = m.group(3)

            channels.append({
                'filename': parts[0],
                'format': fmt,
                'gain': gain,
                'baseline': baseline,
                'units': units,
                'lead': lead_name
            })

        for line in lines:
            if line.startswith('#'):
                c = line.lstrip('#').strip()
                if c:
                    comments.append(c)

        dur_sec = n_samples / fs if fs > 0 else 0
        self.header = {
            'record_name': self.record_name,
            'num_signals': n_sig,
            'sampling_rate': fs,
            'num_samples': n_samples,
            'start_time': start_time,
            'duration_seconds': dur_sec,
            'duration_hours': dur_sec / 3600.0,
            'channels': channels,
            'comments': comments
        }
        return self.header

    def load_signals(self, start_sample: int = 0, n_samples: Optional[int] = None, physical: bool = True) -> List[List[float]]:
        """
        Reads WFDB Format 212 binary signal (.dat).
        Format 212: Each 3 bytes pack two 12-bit signed integers (channel 0 and channel 1).
        
        Args:
            start_sample: Starting sample index.
            n_samples: Number of samples to read. If None, reads entire signal.
            physical: If True, scales ADC units to mV ((ADC - baseline) / gain).
        """
        dat_file = f"{self.record_path}.dat"
        if not os.path.exists(dat_file):
            raise FileNotFoundError(f"Signal file not found: {dat_file}")

        total_samples = self.header.get('num_samples', 0)
        if n_samples is None:
            samples_to_read = total_samples - start_sample
        else:
            samples_to_read = min(n_samples, total_samples - start_sample)

        if samples_to_read <= 0:
            return [[], []]

        # In format 212, 2 samples take 3 bytes
        byte_offset = (start_sample * 3) // 2
        bytes_to_read = ((samples_to_read * 3) // 2) + 3

        with open(dat_file, 'rb') as f:
            f.seek(byte_offset)
            data = f.read(bytes_to_read)

        ch0_raw = []
        ch1_raw = []

        for i in range(0, len(data) - 2, 3):
            b0 = data[i]
            b1 = data[i + 1]
            b2 = data[i + 2]

            # Sample 0: lower byte b0, upper nibble from lower 4 bits of b1
            s0 = b0 | ((b1 & 0x0F) << 8)
            if s0 >= 2048:
                s0 -= 4096

            # Sample 1: lower byte b2, upper nibble from upper 4 bits of b1
            s1 = b2 | ((b1 >> 4) << 8)
            if s1 >= 2048:
                s1 -= 4096

            ch0_raw.append(s0)
            ch1_raw.append(s1)

            if len(ch0_raw) >= samples_to_read:
                break

        self.raw_signals = [ch0_raw, ch1_raw]

        if physical and len(self.header.get('channels', [])) >= 2:
            ch0_meta = self.header['channels'][0]
            ch1_meta = self.header['channels'][1]
            g0, b0 = ch0_meta['gain'], ch0_meta['baseline']
            g1, b1 = ch1_meta['gain'], ch1_meta['baseline']

            ch0_phys = [(s - b0) / g0 for s in ch0_raw]
            ch1_phys = [(s - b1) / g1 for s in ch1_raw]
            self.signals = [ch0_phys, ch1_phys]
            return self.signals
        else:
            self.signals = [[float(s) for s in ch0_raw], [float(s) for s in ch1_raw]]
            return self.signals

    def load_annotations(self, ext: str = 'atr') -> List[Dict[str, Any]]:
        """
        Parses WFDB binary annotation files (.atr, .ari, .qrs).
        Decodes beat labels, rhythm markers (e.g. (VFL, (VFIB, (N), and time offsets.
        """
        ann_file = f"{self.record_path}.{ext}"
        if not os.path.exists(ann_file):
            return []

        with open(ann_file, 'rb') as f:
            data = f.read()

        idx = 0
        curr_sample = 0
        fs = self.header.get('sampling_rate', 250.0)
        annotations = []

        while idx < len(data) - 1:
            b0 = data[idx]
            b1 = data[idx + 1]
            idx += 2

            word = b0 | (b1 << 8)
            code = word >> 10
            delta = word & 0x03FF

            if code == 59:  # SKIP code
                if idx + 4 <= len(data):
                    w1 = data[idx] | (data[idx + 1] << 8)
                    w2 = data[idx + 2] | (data[idx + 3] << 8)
                    idx += 4
                    # WFDB PDP-11 / middle-endian format: (w1 << 16) | w2
                    skip_samples = (w1 << 16) | w2
                    curr_sample += skip_samples
            elif code == 63:  # AUX string code
                str_len = delta
                aux_bytes = data[idx:idx + str_len]
                aux_str = aux_bytes.decode('latin1', errors='ignore').strip('\x00').strip()
                idx += str_len
                if str_len % 2 != 0:
                    idx += 1  # word boundary alignment
                annotations.append({
                    'sample': curr_sample,
                    'time_seconds': round(curr_sample / fs, 3),
                    'code': code,
                    'type': 'AUX',
                    'text': aux_str
                })
            elif code == 0:  # End of annotation file
                break
            else:
                curr_sample += delta
                annotations.append({
                    'sample': curr_sample,
                    'time_seconds': round(curr_sample / fs, 3),
                    'code': code,
                    'type': f'CODE_{code}',
                    'text': ''
                })

        self.annotations = annotations
        return self.annotations

    def summary(self) -> Dict[str, Any]:
        """Returns a high-level summary dictionary of the record."""
        return {
            'record': self.record_name,
            'sampling_rate_hz': self.header.get('sampling_rate'),
            'duration_hours': round(self.header.get('duration_hours', 0), 2),
            'leads': [c['lead'] for c in self.header.get('channels', [])],
            'comments': self.header.get('comments', []),
            'annotations_count': len(self.annotations)
        }


def load_ecg(record_path: str, n_samples: Optional[int] = 1000, physical: bool = True) -> ECGRecord:
    """Convenience helper to quickly load an ECG record and initial samples."""
    rec = ECGRecord(record_path)
    rec.load_signals(n_samples=n_samples, physical=physical)
    # Try loading .atr, or .ari if .atr not present
    if os.path.exists(f"{rec.record_path}.atr"):
        rec.load_annotations('atr')
    elif os.path.exists(f"{rec.record_path}.ari"):
        rec.load_annotations('ari')
    return rec


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 utils/ecg_loader.py <record_path_without_ext> [--samples N]")
        print("Example: python3 utils/ecg_loader.py data/vfdb/418 --samples 500")
        sys.exit(1)

    rec_path = sys.argv[1]
    n_samp = 1000
    if '--samples' in sys.argv:
        pos = sys.argv.index('--samples')
        if pos + 1 < len(sys.argv):
            n_samp = int(sys.argv[pos + 1])

    record = load_ecg(rec_path, n_samples=n_samp)
    print("=" * 60)
    print(f"Record: {record.record_name}")
    print(f"Sampling Frequency: {record.header['sampling_rate']} Hz")
    print(f"Total Duration: {record.header['duration_hours']:.2f} hours")
    print(f"Channels: {[c['lead'] for c in record.header['channels']]}")
    if record.header['comments']:
        print(f"Clinical Info: {record.header['comments']}")
    print(f"Loaded {len(record.signals[0])} samples.")
    print(f"Channel 0 (first 5 samples mV): {record.signals[0][:5]}")
    print(f"Channel 1 (first 5 samples mV): {record.signals[1][:5]}")
    if record.annotations:
        print(f"First annotations ({min(5, len(record.annotations))}):")
        for a in record.annotations[:5]:
            print(f"  Sample {a['sample']} ({a['time_seconds']}s): {a['type']} {a['text']}")
    print("=" * 60)
